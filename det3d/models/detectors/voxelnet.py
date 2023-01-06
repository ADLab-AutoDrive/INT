import os
import numpy as np

from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch
from torch import nn
import torch.nn.functional as F
from ..utils import build_norm_layer
from det3d.models.utils import Empty, GroupNorm, Sequential

from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class FusionModule(nn.Module):
    def __init__(
            self,
            fusion_method: dict,
            norm_cfg=None,
            **kwargs,
    ):
        super(FusionModule, self).__init__()
        self.fusion_method = fusion_method
        cur_pre_fusion_conv = fusion_method["cur_pre_fusion_conv"]
        past_pre_fusion_conv = fusion_method["past_pre_fusion_conv"]
        post_fusion_conv = fusion_method["post_fusion_conv"]
        spatial_weight_conv = fusion_method["spatial_weight_conv"]

        nsweeps = fusion_method.get("nsweeps", None)
        input_channels = fusion_method.get("input_channels", None)

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        self.past_pre_fusion_conv = self._make_layer(**past_pre_fusion_conv)[0] if past_pre_fusion_conv else None
        self.cur_pre_fusion_conv = self._make_layer(**cur_pre_fusion_conv)[0] if cur_pre_fusion_conv else None
        self.post_fusion_conv = self._make_layer(**post_fusion_conv)[0] if post_fusion_conv else None

        if self.fusion_method["type"] in ["add", "add_timeFlag"]:
            if spatial_weight_conv is not None:
                self.spatial_weight_conv = self._make_layer(**spatial_weight_conv)[0]
                self.spatial_weight_conv.add(nn.Sigmoid())
            else:
                self.spatial_weight_conv = None

            if self.fusion_method["sub_type"] == "learned":
                # learned weights.
                self.temporal_weight = nn.Embedding(nsweeps, input_channels)
                nn.init.constant_(self.temporal_weight.weight, 1.0 / nsweeps)

        if self.fusion_method["type"] == "infinite_add":
            if self.fusion_method["sub_type"] == "learned":
                # learned weights.
                self.spatial_temporal_weight = torch.nn.Parameter(torch.zeros((2, input_channels + 1, 512, 512)),
                                                                  requires_grad=True)  # TODO HW make more flexible
                self.spatial_temporal_weight.data.fill_(1 / 2.)

        if self.fusion_method["type"] == "infinite_GRU":
            reset_gate_conv = fusion_method["reset_gate_conv"]
            update_gate_conv = fusion_method["update_gate_conv"]
            candidate_conv = fusion_method["candidate_conv"]
            self.reset_gate_conv = self._make_layer(**reset_gate_conv)[0]  # conv+bn+sigmoid
            self.update_gate_conv = self._make_layer(**update_gate_conv)[0]  # conv+bn+sigmoid
            self.candidate_conv = self._make_layer(**candidate_conv)[0]  # conv+bn+tanh

    def _make_layer(self, inplanes, planes, num_blocks, stride=1, kernel_size=3, activation=None):
        if activation is None:
            activation = nn.ReLU()
        assert kernel_size in [1, 3]
        padding = 0 if kernel_size == 1 else 1
        block = Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            activation,
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, kernel_size, padding=padding, bias=False))
            block.add(
                build_norm_layer(self._norm_cfg, planes)[1],
                # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
            block.add(activation)

        return block, planes

    def forward(self, x, past_x_list):
        if self.cur_pre_fusion_conv is not None:
            x = self.cur_pre_fusion_conv(x)
        if self.past_pre_fusion_conv is not None:
            past_x_list = [self.past_pre_fusion_conv(p_x) for p_x in past_x_list]

        infinite_feat_map = None

        if self.fusion_method.type == "concat":
            fusion_x = torch.cat([x] + past_x_list, dim=1)
        elif self.fusion_method.type == "infinite_add":
            assert len(past_x_list) == 1, len(past_x_list)
            past_x = past_x_list[0]
            if self.fusion_method.sub_type == "weighted":
                if past_x.sum() == 0:
                    fusion_x = x
                else:
                    fusion_x = 0.5 * (x + past_x)
            elif self.fusion_method.sub_type == "learned":
                B = x.shape[0]
                st_weight = self.spatial_temporal_weight.unsqueeze(1).repeat(1, B, 1, 1, 1)
                fusion_x = x * st_weight[0] + past_x * st_weight[1]
                # print(self.spatial_temporal_weight.max())

        elif self.fusion_method.type == "add":
            if self.spatial_weight_conv is None:
                B, _, H, W = x.shape
                spatial_weight = torch.zeros((B, H, W), device=x.device, requires_grad=False, dtype=x.dtype)
                x_cp = x.detach().clone()
                spatial_weight[x_cp.sum(dim=1) != 0] = 1
                spatial_weight = spatial_weight.unsqueeze(1)
            else:
                spatial_weight = self.spatial_weight_conv(x)
            if self.fusion_method.sub_type == "weighted":
                all_x = [x] + past_x_list
                # weights = [1 / len(all_x) for _ in range(len(all_x))]  # uniform is not good!
                weights = 1. / np.arange(1, len(all_x) + 1)
                weights = weights / weights.sum()
                fusion_x_list = [w * xx for w, xx in zip(weights, all_x)]
                fusion_x = fusion_x_list[0]
                for ele in fusion_x_list[1:]:
                    fusion_x += ele
            elif self.fusion_method.sub_type == "exp":
                all_x = [x] + past_x_list
                weights = 1 / np.power(2, np.arange(1, len(all_x) + 1))
                weights = weights / weights.sum()
                fusion_x_list = [w * xx for w, xx in zip(weights, all_x)]
                fusion_x = fusion_x_list[0]
                for ele in fusion_x_list[1:]:
                    fusion_x += ele
            elif self.fusion_method.sub_type == "learned":
                B, C, H, W = x.shape
                fusion_x = x * self.temporal_weight.weight[0].reshape(1, -1, 1, 1).repeat(B, 1, H, W)
                for i, p_x in enumerate(past_x_list):
                    fusion_x += p_x * self.temporal_weight.weight[i + 1].reshape(1, -1, 1, 1).repeat(B, 1, H, W)
            else:
                raise NotImplementedError

            fusion_x = torch.cat([spatial_weight, fusion_x], dim=1)  # C+1
        elif self.fusion_method.type == "add_timeFlag":
            if self.spatial_weight_conv is not None:
                spatial_weight = self.spatial_weight_conv(x)
            else:
                spatial_weight = None

            all_x = [x] + past_x_list
            with torch.no_grad():
                B, _, H, W = x.shape
                time_flag_list = []
                for one_x in all_x:
                    time_flag = torch.zeros((B, H, W), device=x.device)
                    mask = (one_x.sum(dim=1) != 0)
                    time_flag[mask] = 1
                    time_flag = time_flag.unsqueeze(dim=1)
                    time_flag_list.append(time_flag)

            all_x = [torch.cat([xx, tt], dim=1) for xx, tt in zip(all_x, time_flag_list)]

            if self.fusion_method.sub_type == "weighted":
                # all_x = [x] + past_x_list
                # weights = [1 / len(all_x) for _ in range(len(all_x))]  # uniform is not good!
                weights = 1. / np.arange(1, len(all_x) + 1)
                weights = weights / weights.sum()
                fusion_x_list = [w * xx for w, xx in zip(weights, all_x)]
                fusion_x = sum(fusion_x_list)
            elif self.fusion_method.sub_type == "exp":
                # all_x = [x] + past_x_list
                weights = 1 / np.power(2, np.arange(1, len(all_x) + 1))
                weights = weights / weights.sum()
                fusion_x_list = [w * xx for w, xx in zip(weights, all_x)]
                fusion_x = sum(fusion_x_list)
            else:
                raise NotImplementedError

            if spatial_weight is not None:
                fusion_x = torch.cat([spatial_weight, fusion_x], dim=1)  # C+1
        elif self.fusion_method.type == "infinite_max":
            assert len(past_x_list) == 1, len(past_x_list)
            past_x = past_x_list[0]
            past_feat, past_time = past_x[:, 0:-1, ...], past_x[:, -1, ...]
            assert past_feat.shape[1] == x.shape[1]

            past_mask = (past_feat.sum(dim=1) != 0)
            past_time[past_mask] += 0.05

            concat_x = torch.cat([past_feat.unsqueeze(0), x.unsqueeze(0)], dim=0)  # 2xBxCxHxW
            fusion_x = torch.max(concat_x, 0)[0]  # BxCxHxW

            fusion_x = torch.cat([fusion_x, past_time.unsqueeze(1)], dim=1)
            infinite_feat_map = fusion_x.detach()

            # cur_valid grid mask
            B, _, H, W = x.shape
            cur_mask = torch.zeros((B, H, W), device=x.device, requires_grad=False, dtype=x.dtype)
            x_cp = x.detach().clone()
            cur_mask[x_cp.sum(dim=1) != 0] = 1
            fusion_x = torch.cat([fusion_x, cur_mask.unsqueeze(1)], dim=1)

        elif self.fusion_method.type == "infinite_GRU":
            assert len(past_x_list) == 1, len(past_x_list)
            past_x = past_x_list[0]
            past_feat, past_time = past_x[:, 0:-1, ...], past_x[:, -1, ...]
            assert past_feat.shape[1] == x.shape[1]

            past_mask = (past_feat.sum(dim=1) != 0)
            past_time[past_mask] += 0.05

            # GRU gate
            reset_weight = self.reset_gate_conv(torch.cat([x, past_feat], dim=1))  # conv+bn+sigmoid
            update_weight = self.update_gate_conv(torch.cat([x, past_feat], dim=1))  # conv+bn+sigmoid
            candidate_fusion_x = self.candidate_conv(torch.cat([x, reset_weight * past_feat], dim=1))  # conv+bn+tanh

            fusion_x = update_weight * past_feat + (1 - update_weight) * candidate_fusion_x

            fusion_x = torch.cat([fusion_x, past_time.unsqueeze(1)], dim=1)
            infinite_feat_map = fusion_x.detach()

            # cur_valid grid mask
            with torch.no_grad():
                B, _, H, W = x.shape
                spatial_mask = torch.zeros((B, H, W), device=x.device, requires_grad=False, dtype=x.dtype)
                spatial_mask[(x.sum(dim=1) != 0)] = 1
            fusion_x = torch.cat([fusion_x, spatial_mask.unsqueeze(1)], dim=1)

        if self.post_fusion_conv is not None:
            fusion_x = self.post_fusion_conv(fusion_x)

        return fusion_x, infinite_feat_map


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
            self,
            batch_size,
            reader,
            backbone,
            neck,
            bbox_head,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            fm_fusion_method=None,
            hm_fusion_method=None,
            pc_fusion_method=None,
            voxelization=None,
            **kwargs,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, voxelization, init_inside=False,
        )
        if kwargs.get("freeze_backbone", False):
            self.backbone = self.backbone.freeze()

        if fm_fusion_method is not None:
            self.fm_fusion_method = fm_fusion_method
            self.fm_fusion_module = FusionModule(fm_fusion_method)

        if hm_fusion_method is not None:
            self.hm_fusion_method = hm_fusion_method
            self.hm_fusion_module = FusionModule(hm_fusion_method)

        self.batch_size = batch_size

        if pc_fusion_method is not None:
            self.pc_fusion_method = pc_fusion_method
            assert self.batch_size is not None
            max_pc = pc_fusion_method['max_points_total']
            pc_chan = pc_fusion_method['pc_channel']
            self.register_buffer('infinite_pc', torch.zeros((self.batch_size, max_pc, pc_chan),
                                                            dtype=torch.float))

            assert kwargs.get("freeze_backbone", False) is False, 'when use pcFusion, PFN should not be frozen!'

        if fm_fusion_method is not None:
            w, h = fm_fusion_method.fm_WH
            c = fm_fusion_method.input_channels
            if fm_fusion_method.type in ["infinite_add", "infinite_max", "infinite_GRU"]:
                self.register_buffer('infinite_feat_map', torch.zeros((self.batch_size, c + 1, h, w),
                                                                      dtype=torch.float))
            else:
                self.register_buffer('infinite_feat_map', torch.zeros((self.batch_size, c, h, w),
                                                                      dtype=torch.float))

        if hm_fusion_method is not None:
            w, h, c = hm_fusion_method.hm_WHC
            self.register_buffer('infinite_heat_map', torch.zeros((self.batch_size, c, h, w),
                                                                  dtype=torch.float))

        self.init_weights(pretrained=pretrained)

    def forward(self, example, return_loss=True, **kwargs):
        RETURN_BEV_FEAT = kwargs.pop('RETURN_BEV_FEAT', False)

        cur_points = example['points']
        batch_size = cur_points[-1, 0].int().item() + 1

        ## pc_fusion
        if getattr(self, 'pc_fusion_method', None):
            past_points = []
            for bs_cnt, one_pc in enumerate(self.infinite_pc):
                valid_pc = one_pc[one_pc.sum(-1) != 0]
                bs_col = valid_pc.new_ones(size=(valid_pc.shape[0], 1)) * bs_cnt
                valid_pc = torch.cat([bs_col, valid_pc], dim=1)
                past_points.append(valid_pc)

            # add time channel
            time_chan = cur_points.new_zeros(size=(cur_points.shape[0], 1))
            cur_points = torch.cat([cur_points, time_chan], dim=-1)
            points = torch.cat([cur_points] + past_points, dim=0)
        else:
            points = cur_points

        if self.reader_type == 'DynamicVoxelEncoder':
            output = self.reader(points, batch_size)
            voxels, coors, shape = output

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=batch_size,
                input_shape=shape,
            )
            input_features = voxels
        else:
            if self.with_voxelization:
                voxels, num_points_in_voxel, coordinates, input_shape = \
                    self.voxelization(points, batch_size)
            else:
                voxels = example["voxels"]
                coordinates = example["coordinates"]
                num_points_in_voxel = example["num_points"]
                input_shape = example["shape"][0]

            data = dict(
                features=voxels,
                num_voxels=num_points_in_voxel,
                coors=coordinates,
                batch_size=batch_size,
                input_shape=input_shape,
            )

            input_features = self.reader(data["features"], data["num_voxels"])

        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        if getattr(self, 'fm_fusion_method', None):
            if self.fm_fusion_method.type == "infinite_add":
                with torch.no_grad():
                    B, _, H, W = x.shape
                    time_flag = torch.zeros((B, H, W), device=x.device)
                    mask = (x.detach().sum(dim=1) != 0)
                    time_flag[mask] = 1
                    time_flag = time_flag.unsqueeze(dim=1)
                x = torch.cat([x, time_flag], dim=1)
            x, infinite_feat_map = self.fm_fusion_module(x, [self.infinite_feat_map])
            if infinite_feat_map is None:
                self.infinite_feat_map = x.detach()
            else:
                self.infinite_feat_map = infinite_feat_map

        if self.with_neck:
            x = self.neck(x)

        bev_feat = x

        if getattr(self, 'hm_fusion_method', None):
            preds, final_feat = self.bbox_head(x, past_preds=self.infinite_heat_map, points=cur_points)
        else:
            preds, final_feat = self.bbox_head(x, points=cur_points)

        if getattr(self, 'hm_fusion_method', None):
            with torch.no_grad():
                cur_hm = torch.cat([pred['hm'].detach() for pred in preds], dim=1)
                if self.hm_fusion_method.get('use_reg', False):
                    # reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2), 'vel'
                    for reg_type in ['reg', 'height', 'dim', 'rot', 'vel', 'iou']:
                        if not reg_type in preds[0]:
                            continue
                        cur_hm = torch.cat([cur_hm] + [pred[reg_type].detach() for pred in preds], dim=1)

                fusion_hm, infinite_heat_map = self.hm_fusion_module(cur_hm, [self.infinite_heat_map])
                self.infinite_heat_map = fusion_hm if infinite_heat_map is None else infinite_heat_map

        new_preds = self.preds_copy(preds)
        boxes_list = self.bbox_head.predict(example, new_preds, self.test_cfg)
        points_seg_score = torch.sigmoid(
            self.bbox_head.point_seg_logits.view(-1)) if self.bbox_head.use_point_seg else None
        if getattr(self, 'pc_fusion_method', None):
            with torch.no_grad():
                self.pc_fusion(batch_size, cur_points, boxes_list, points_seg_score)

        if return_loss and RETURN_BEV_FEAT:  # used for two stage train
            return self.bbox_head.loss(example, preds, self.test_cfg), boxes_list, bev_feat, final_feat, voxel_feature
        elif return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        elif RETURN_BEV_FEAT:  # used for two stage test
            return None, boxes_list, bev_feat, final_feat, voxel_feature
        else:
            return boxes_list

    def pc_fusion(self, batch_size, cur_points, boxes_list=None, points_seg_score=None):
        max_points_per_frame = self.pc_fusion_method['max_points_per_frame']
        min_points_per_frame = self.pc_fusion_method['min_points_per_frame']
        max_time_limit = self.pc_fusion_method['max_time_limit']
        max_points_total = self.pc_fusion_method['max_points_total']
        time_interval = self.pc_fusion_method['time_interval']

        if self.pc_fusion_method['keep_strategy'] == 'point_seg':
            point_seg_thre = self.pc_fusion_method['point_seg_thre']
            for bs_cnt in range(batch_size):
                bs_mask = (cur_points[:, 0] == bs_cnt)
                one_points = cur_points[bs_mask][:, 1:]
                one_seg_score = points_seg_score[bs_mask]
                keep_idxs = (one_seg_score > point_seg_thre).nonzero().view(-1)

                if len(keep_idxs) > max_points_per_frame:
                    tmp = torch.randperm(len(keep_idxs))[:max_points_per_frame]
                    keep_idxs = keep_idxs[tmp]

                one_past_pc = self.infinite_pc[bs_cnt].clone()
                past_valid_mask = (one_past_pc.sum(1) != 0) & (one_past_pc[:, -1] < max_time_limit)
                past_valid_idxs = past_valid_mask.nonzero().view(-1)

                if (max_points_total - len(past_valid_idxs)) < len(keep_idxs):
                    past_keep_num = max_points_total - len(keep_idxs)
                    tmp = torch.randperm(len(past_valid_idxs))[:past_keep_num]
                    past_valid_idxs = past_valid_idxs[tmp]

                self.infinite_pc[bs_cnt][...] = 0
                self.infinite_pc[bs_cnt][0:len(past_valid_idxs)] = one_past_pc[past_valid_idxs]
                valid_len = len(past_valid_idxs) + len(keep_idxs)
                self.infinite_pc[bs_cnt][len(past_valid_idxs):valid_len] = one_points[keep_idxs]
                self.infinite_pc[bs_cnt][:valid_len, -1] += time_interval

        elif self.pc_fusion_method['keep_strategy'] == 'in_box':
            for bs_cnt in range(batch_size):
                boxes3d = boxes_list[bs_cnt]['box3d_lidar'].detach().clone()
                boxes3d = boxes3d[:, [0, 1, 2, 4, 3, 5, -1]]
                one_points = cur_points[cur_points[:, 0] == bs_cnt][:, 1:]

                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    one_points[:, 0:3].unsqueeze(dim=0), boxes3d[:, 0:7].unsqueeze(0).contiguous()
                ).long().squeeze(dim=0)

                keep_idxs = (box_idxs_of_pts != -1).nonzero().view(-1)
                if len(keep_idxs) > max_points_per_frame:
                    tmp = torch.randperm(len(keep_idxs))[:max_points_per_frame]
                    keep_idxs = keep_idxs[tmp]
                # if len(keep_idxs) < min_points_per_frame:
                #     more_idxs = torch.arange(min_points_per_frame).to(keep_idxs.device)
                #     keep_idxs = torch.cat([keep_idxs, more_idxs])

                one_past_pc = self.infinite_pc[bs_cnt].clone()
                past_valid_mask = (one_past_pc.sum(1) != 0) & (one_past_pc[:, -1] < max_time_limit)
                past_valid_idxs = past_valid_mask.nonzero().view(-1)

                if (max_points_total - len(past_valid_idxs)) < len(keep_idxs):
                    past_keep_num = max_points_total - len(keep_idxs)
                    tmp = torch.randperm(len(past_valid_idxs))[:past_keep_num]
                    past_valid_idxs = past_valid_idxs[tmp]

                self.infinite_pc[bs_cnt][...] = 0
                self.infinite_pc[bs_cnt][0:len(past_valid_idxs)] = one_past_pc[past_valid_idxs]
                valid_len = len(past_valid_idxs) + len(keep_idxs)
                self.infinite_pc[bs_cnt][len(past_valid_idxs):valid_len] = one_points[keep_idxs]
                self.infinite_pc[bs_cnt][:valid_len, -1] += time_interval

    def preds_copy(self, preds):
        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach().clone()
            new_preds.append(new_pred)
        return new_preds

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        kwargs.update(dict(RETURN_BEV_FEAT=True))
        one_stage_loss, boxes, bev_feature, final_feat, voxel_feature = self.forward(example, return_loss=return_loss,
                                                                                     **kwargs)
        if return_loss:
            return boxes, bev_feature, voxel_feature, final_feat, one_stage_loss
        else:
            return boxes, bev_feature, voxel_feature, final_feat, None
