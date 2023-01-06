import torch.nn as nn

from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from ..utils.finetune_utils import FrozenBatchNorm2d
from det3d.torchie.trainer import load_checkpoint


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    def __init__(
            self,
            reader,
            backbone,
            neck=None,
            bbox_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            voxelization=None,
            init_inside=True,
    ):
        super(SingleStageDetector, self).__init__()
        if voxelization is not None:
            self.voxelization = builder.build_voxelization(voxelization)
        self.reader = builder.build_reader(reader)
        self.reader_type = reader['type']
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if init_inside:
            self.init_weights(pretrained=pretrained)
        else:
            print("attention!! init outside!!")

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))

    def extract_feat(self, data):
        input_features = self.reader(data)
        x = self.backbone(input_features)
        if self.with_neck:
            x = self.neck(x)
        return x

    def aug_test(self, example, rescale=False):
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self
