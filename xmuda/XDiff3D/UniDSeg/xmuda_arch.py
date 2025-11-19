import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from xmuda.XDiff3D.UniDSeg.seg_clip import SegCLIP
from xmuda.XDiff3D.spconv_unet_v1m1_base import SpUNetBase
from ..diffusion_extractor import DiffusionTrainer_

class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 ):
        super(Net2DSeg, self).__init__()

        # 2D network
        if backbone_2d == 'ViT-B-16' or backbone_2d == 'ViT-L-14' or backbone_2d == 'SAM_ViT-L':
            self.net_2d = SegCLIP(backbone_2d)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        self.linear = nn.Linear(feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(feat_channels, num_classes)

        self.diff = DiffusionTrainer_('stable-diffusion-2')

    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        depth = data_batch['depth']
        img_indices = data_batch['img_indices']

        x = self.net_2d(img, depth)
        object_query_2D, object_query_2D_de=self.diff.feature_extraction(img, [0])

        # 2D-3D feature lifting
        img_feats = []
        num_points = []
        for i in range(x.shape[0]):
            proj_feats = x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]]
            img_feats.append(proj_feats)

            num_point = proj_feats.shape[0]
            num_points.append(num_point)
        img_feats = torch.cat(img_feats, 0)

        x = self.linear(img_feats)



        preds = {
            'feats': img_feats,
            'seg_logit': x,
            'num_points': num_points,
        }

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(img_feats)

        return preds,object_query_2D,object_query_2D_de



class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = SpUNetBase(in_channels=1, num_classes=num_classes)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # 2nd segmentation head
        self.dual_head = dual_head

    def forward_delta_feat(self, feats, tokens):
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f

    def forward(self, data_batch,object_query_2D):
        inter_feat_3d, object_query_3D = self.net_3d.encoder_forward(data_batch['x'])

        delta_object_query_3D = self.forward_delta_feat(
            object_query_3D,
            object_query_2D,
        )
        delta_object_query_3D = delta_object_query_3D * self.scale
        object_query_3D_diff = object_query_3D + delta_object_query_3D


        x1, x2, x_ori, x_refine = self.net_3d.decoder_forward(inter_feat_3d, object_query_3D, object_query_3D_diff)

        preds = {
            'seg_logit': x1,
        }

        if self.dual_head:
            preds['seg_logit2'] = x2

        return preds, object_query_3D, x_ori, x_refine

