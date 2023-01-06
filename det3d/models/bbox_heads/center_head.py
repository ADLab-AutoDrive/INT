# ------------------------------------------------------------------------------
# Portions of this code are from
# det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)
# Copyright (c) 2019 朱本金
# Licensed under the MIT License
# ------------------------------------------------------------------------------

import logging
from collections import defaultdict
from det3d.core import box_torch_ops
import torch
import torch.nn.functional as F
from det3d.torchie.cnn import kaiming_init
from torch import double, nn
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss
from det3d.models.utils import Sequential
from det3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from ...core.utils.center_utils import _transpose_and_gather_feat
from ..registry import HEADS
import copy

try:
    from det3d.ops.dcn import DeformConv
except:
    print("Deformable Convolution not built!")

from det3d.core.utils.circle_nms_jit import circle_nms


def make_fc_layers(fc_cfg, input_channels, output_channels):
    fc_layers = []
    c_in = input_channels
    for k in range(0, fc_cfg.__len__()):
        fc_layers.extend([
            nn.Linear(c_in, fc_cfg[k], bias=False),
            nn.BatchNorm1d(fc_cfg[k]),
            nn.ReLU(),
        ])
        c_in = fc_cfg[k]
    fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
    return nn.Sequential(*fc_layers)


def batch_grid_sample(bev_images, points, pc_range, mode='bilinear'):
    '''
    Args:
        bev_images: (B, C, H, W)
        pxpy: NxC, first 3 dim is (bs, x, y...)
        pc_range: [-x, -y, -z, x, y, z]
        xyz: Nx3
    Returns:
        points feature: (N, C)
    '''
    # pxpy: Nx3, 3=(bs_idx, px, py) normed, (-1, 1)
    pxpy = points[:, :3].clone()
    pxpy[:, 1] = (pxpy[:, 1] - pc_range[0]) / (pc_range[3] - pc_range[0])  # [0, 1]
    pxpy[:, 1] = pxpy[:, 1] * 2 - 1  # [-1, 1]
    pxpy[:, 2] = (pxpy[:, 2] - pc_range[1]) / (pc_range[4] - pc_range[1])  # [0, 1]
    pxpy[:, 2] = pxpy[:, 2] * 2 - 1  # [-1, 1]

    bs = bev_images.shape[0]
    feature_list = []
    for bs_idx in range(bs):
        mask = (pxpy[:, 0] == bs_idx)
        pxpy_single = pxpy[mask][:, 1:3]  # Nx2
        pxpy_single = pxpy_single.unsqueeze(0).unsqueeze(0)  # 1x1xNx2
        sampled_features = F.grid_sample(bev_images[bs_idx:(bs_idx + 1)], pxpy_single, mode=mode)  # 1xCx1xN
        sampled_features = sampled_features.squeeze(0).squeeze(1).transpose(1, 0).contiguous()  # NxC
        feature_list.append(sampled_features)
    return torch.cat(feature_list, dim=0)  # (N1 + N2 + N3 + ..., C)


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()

    def forward(self, x, ):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x


class SepHead(nn.Module):
    def __init__(
            self,
            in_channels,
            heads,
            head_conv=64,
            final_kernel=1,
            bn=False,
            init_bias=-2.19,
            **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(nn.Conv2d(in_channels, head_conv,
                                 kernel_size=final_kernel, stride=1,
                                 padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                             kernel_size=final_kernel, stride=1,
                             padding=final_kernel // 2, bias=True))

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class DCNSepHead(nn.Module):
    def __init__(
            self,
            in_channels,
            num_cls,
            heads,
            head_conv=64,
            final_kernel=1,
            bn=False,
            init_bias=-2.19,
            **kwargs,
    ):
        super(DCNSepHead, self).__init__(**kwargs)

        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)

        self.feature_adapt_reg = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)

        # heatmap prediction head 
        self.cls_head = Sequential(
            nn.Conv2d(in_channels, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls,
                      kernel_size=3, stride=1,
                      padding=1, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(init_bias)

        # other regression target 
        self.task_head = SepHead(in_channels, heads, head_conv=head_conv, bn=bn, final_kernel=final_kernel)

    def forward(self, x):
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret


@HEADS.register_module
class CenterHead(nn.Module):
    def __init__(
            self,
            in_channels=[128, ],
            tasks=[],
            dataset='nuscenes',
            weight=0.25,
            code_weights=[],
            common_heads=dict(),
            logger=None,
            init_bias=-2.19,
            share_conv_channel=64,
            num_hm_conv=2,
            dcn_head=False,
            hm_fusion_method=None,
            iou_weight=0,
            point_seg=None,
    ):
        super(CenterHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()
        self.iou_weight = iou_weight

        self.box_n_dim = 9 if 'vel' in common_heads else 7
        self.use_direction_classifier = False

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

        # a shared convolution 
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        task_input_channels = share_conv_channel
        if hm_fusion_method is not None:
            task_input_channels += hm_fusion_method['hm_WHC'][-1]

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(task_input_channels, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )
            else:
                self.tasks.append(
                    DCNSepHead(task_input_channels, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )

        self.use_point_seg = point_seg is not None
        if self.use_point_seg:
            self.point_mlps = nn.ModuleList()
            self.point_mlps.append( #TODO this layer no bn, use bias, not right.
                make_fc_layers(point_seg['point_lift_mlp'], point_seg['point_features'], share_conv_channel))
            self.point_mlps.append(
                make_fc_layers(point_seg['point_head_mlp'], share_conv_channel + share_conv_channel, 1))

            self.pc_range = point_seg['pc_range']
            self.point_seg_logits = None
            self.point_seg_loss_weight = point_seg['point_seg_loss_weight']

        logger.info("Finish CenterHead Initialization")

    def forward(self, x, **kwargs):
        ret_dicts = []

        x = self.shared_conv(x)

        if self.use_point_seg:
            points = kwargs.get('points')
            point_feat = self.point_mlps[0](points[:, 1:])
            bev_feat = batch_grid_sample(x, points, self.pc_range)
            p_feat = torch.cat([point_feat, bev_feat], dim=1)
            p_seg_logits = self.point_mlps[1](p_feat)
            # ret_dicts.append({'point_seg': p_seg_logits})
            self.point_seg_logits = p_seg_logits

        past_preds = kwargs.get('past_preds', None)
        if past_preds is not None:
            x = torch.cat([x, past_preds], dim=1)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts, x

    @torch.no_grad()
    def _iou_target(self, example, preds_dict, task_id, test_cfg):
        batch, _, H, W = preds_dict['hm'].size()
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, 1, H, W).repeat(batch, 1, 1, 1).to(preds_dict['hm'])
        xs = xs.view(1, 1, H, W).repeat(batch, 1, 1, 1).to(preds_dict['hm'])

        batch_det_dim = torch.exp(preds_dict['dim'])
        batch_det_rots = preds_dict['rot'][:, 0:1, :, :]
        batch_det_rotc = preds_dict['rot'][:, 1:2, :, :]
        batch_det_reg = preds_dict['reg']
        batch_det_hei = preds_dict['height']
        batch_det_rot = torch.atan2(batch_det_rots, batch_det_rotc)
        batch_det_xs = xs + batch_det_reg[:, 0:1, :, :]
        batch_det_ys = ys + batch_det_reg[:, 1:2, :, :]
        batch_det_xs = batch_det_xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
        batch_det_ys = batch_det_ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]
        # (B, 7, H, W)
        batch_box_preds = torch.cat([batch_det_xs, batch_det_ys, batch_det_hei, batch_det_dim, batch_det_rot], dim=1)

        batch_box_preds = _transpose_and_gather_feat(batch_box_preds, example['ind'][task_id])

        target_box = example['anno_box'][task_id]
        batch_gt_dim = torch.exp(target_box[..., 3:6])
        batch_gt_reg = target_box[..., 0:2]
        batch_gt_hei = target_box[..., 2:3]
        batch_gt_rot = torch.atan2(target_box[..., -2:-1], target_box[..., -1:])
        batch_gt_xs = _transpose_and_gather_feat(xs, example['ind'][task_id]) + batch_gt_reg[..., 0:1]
        batch_gt_ys = _transpose_and_gather_feat(ys, example['ind'][task_id]) + batch_gt_reg[..., 1:2]
        batch_gt_xs = batch_gt_xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
        batch_gt_ys = batch_gt_ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]
        # (B, max_obj, 7)
        batch_box_targets = torch.cat([batch_gt_xs, batch_gt_ys, batch_gt_hei, batch_gt_dim, batch_gt_rot], dim=-1)

        iou_targets = boxes_iou3d_gpu(batch_box_preds.reshape(-1, 7), batch_box_targets.reshape(-1, 7))[range(
            batch_box_preds.reshape(-1, 7).shape[0]), range(batch_box_targets.reshape(-1, 7).shape[0])]

        return iou_targets.reshape(batch, -1, 1)

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

    def loss(self, example, preds_dicts, test_cfg=None, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id],
                                example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.dataset in ['waymo', 'nuscenes']:
                if 'vel' in preds_dict:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['vel'], preds_dict['rot']), dim=1)
                else:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['rot']), dim=1)
                    target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]]  # remove vel target
            else:
                raise NotImplementedError()

            ret = {}

            # Regression loss for dimension, offset, height, rotation            
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id],
                                     target_box)

            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight * loc_loss

            if self.iou_weight > 0:
                iou_targets = self._iou_target(example, preds_dict, task_id, test_cfg)
                iou_loss = self.crit_reg(preds_dict['iou'], example['mask'][task_id], example['ind'][task_id],
                                         iou_targets)
                loss += self.iou_weight * iou_loss.sum()
                ret['iou_loss'] = iou_loss.detach().cpu()

            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss': loc_loss,
                        'loc_loss_elem': box_loss.detach().cpu(),
                        'num_positive': example['mask'][task_id].float().sum()})

            rets.append(ret)

        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        if self.use_point_seg:
            points_seg_flat = self.point_seg_logits.view(-1)
            points_seg_label = example['points_seg_label']
            loss_point_seg = F.binary_cross_entropy(torch.sigmoid(points_seg_flat), points_seg_label.float(),
                                                    reduction='none')
            cls_valid_mask = (points_seg_label >= 0).float()
            loss_point_seg = (loss_point_seg * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            # TODO(tmp) add point seg loss length equal to loss list
            rets_merged['loss_point_seg'] = [loss_point_seg] * len(rets_merged['loss'])
            for i in range(len(rets_merged['loss'])):
                rets_merged['loss'][i] += self.point_seg_loss_weight * loss_point_seg

        return rets_merged

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # transform the prediction map back to their original coordinate befor flipping
                    # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
                    # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is 
                    # X and Y flip pointcloud(x=-x, y=-y).
                    # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
                    # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
                    # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1])
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[:4 * int(batch_size):4]

            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']
            if self.iou_weight > 0:
                batch_iou = preds_dict['iou']

            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)

                # y = -y reg_y = 1-reg_y
                batch_reg[:, 1, ..., 1] = 1 - batch_reg[:, 1, ..., 1]
                batch_reg[:, 2, ..., 0] = 1 - batch_reg[:, 2, ..., 0]

                batch_reg[:, 3, ..., 0] = 1 - batch_reg[:, 3, ..., 0]
                batch_reg[:, 3, ..., 1] = 1 - batch_reg[:, 3, ..., 1]
                batch_reg = batch_reg.mean(dim=1)

                # first yflip 
                # y = -y theta = pi -theta
                # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
                # batch_rots[:, 1] the same
                batch_rotc[:, 1] *= -1

                # then xflip x = -x theta = 2pi - theta
                # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
                # batch_rots[:, 2] the same
                batch_rots[:, 2] *= -1

                # double flip 
                batch_rots[:, 3] *= -1
                batch_rotc[:, 3] *= -1

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H * W, 2)
            batch_hei = batch_hei.reshape(batch, H * W, 1)

            batch_rot = batch_rot.reshape(batch, H * W, 1)
            batch_dim = batch_dim.reshape(batch, H * W, 3)
            batch_hm = batch_hm.reshape(batch, H * W, num_cls)

            if self.iou_weight > 0:
                # multiply together for the final score
                batch_iou = torch.clamp(batch_iou.reshape(batch, H * W, 1), min=0, max=1)
                batch_hm = batch_hm * torch.pow(batch_iou, test_cfg.get('cf_weight', 2))

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']

                if double_flip:
                    # flip vy
                    batch_vel[:, 1, ..., 1] *= -1
                    # flip vx
                    batch_vel[:, 2, ..., 0] *= -1

                    batch_vel[:, 3] *= -1

                    batch_vel = batch_vel.mean(dim=1)

                batch_vel = batch_vel.reshape(batch, H * W, 2)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else:
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            if test_cfg.get('per_class_nms', False):
                pass
            else:
                rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range, task_id))

                # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list

    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range, task_id):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                            & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            if test_cfg.get('circular_nms', False):
                centers = boxes_for_nms[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(boxes, min_radius=test_cfg.min_radius[task_id],
                                       post_max_size=test_cfg.nms.nms_post_max_size)
            else:
                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float(), scores.float(),
                                                          thresh=test_cfg.nms.nms_iou_threshold,
                                                          pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                                          post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts


import numpy as np


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep
