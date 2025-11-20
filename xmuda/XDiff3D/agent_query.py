from functools import reduce, partial
from operator import mul
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.models.layers import DropPath  # pip install timm


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):

        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # x: (B, N, C)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor, feats: Tensor, H: int, W: int):
        B, Nq, C = tokens.shape
        Bf, Nf, Cf = feats.shape
        assert B == Bf and C == Cf, \
            f"tokens({tokens.shape}) and feats({feats.shape}) must match in batch and dim"

        q = self.q(tokens).reshape(B, Nq, self.num_heads,
                                   C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_ = feats.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)  # (B, N', C)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(feats).reshape(B, -1, 2, self.num_heads,
                                        C // self.num_heads).permute(
                2, 0, 3, 1, 4).contiguous()

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,Nq,Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, tokens: Tensor, feats: Tensor, H: int, W: int):
        tokens = tokens + self.drop_path(
            self.attn(self.norm1(tokens), feats, H, W)
        )

        B, Nq, C = tokens.shape
        tokens = tokens + self.drop_path(
            self.mlp(self.norm2(tokens), H=1, W=Nq)
        )
        return tokens



class object_query_2D_Dual(nn.Module):
    def __init__(
            self,
            num_layers: int,
            embed_dims: int,
            patch_size: int,
            query_dims: int = 256,
            token_length: int = 100,
            use_softmax: bool = True,
            link_token_to_query: bool = True,
            scale_init: float = 0.001,
            zero_mlp_delta_f: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f

        self.block1 = nn.ModuleList([
            Block(
                dim=1024,
                num_heads=8,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0,
                attn_drop=0,
                drop_path=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                sr_ratio=1) for i in range(12)
        ])

        self.block2 = nn.ModuleList([
            Block(
                dim=1024,
                num_heads=8,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0,
                attn_drop=0,
                drop_path=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                sr_ratio=1) for i in range(12)
        ])

        self.create_model()

    def create_model(self):
        self.learnable_tokens1 = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.learnable_tokens2 = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )

        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )

        nn.init.uniform_(self.learnable_tokens1.data, -val, val)
        nn.init.uniform_(self.learnable_tokens2.data, -val, val)

        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)

        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

    def _process_tokens_to_query(self, tokens):
        tokens = self.transform(tokens).permute(1, 2, 0)
        tokens = torch.cat(
            [
                F.max_pool1d(tokens, kernel_size=self.num_layers),
                F.avg_pool1d(tokens, kernel_size=self.num_layers),
                tokens[:, :, -1].unsqueeze(-1),
            ],
            dim=-1,
        )
        querys = self.merge(tokens.flatten(-2, -1))
        return querys

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens1, tokens2 = self.get_tokens(-1)

            querys1 = self._process_tokens_to_query(tokens1)
            querys2 = self._process_tokens_to_query(tokens2)

            return feats, querys1, querys2
        else:
            return feats

    def get_tokens(self, layer: int):
        if layer == -1:
            return self.learnable_tokens1, self.learnable_tokens2
        else:
            return self.learnable_tokens1[layer], self.learnable_tokens2[layer]

    def _apply_svd(self, x: Tensor, k: int) -> Tensor:
        if x.shape[1] == x.size(0) and x.dim() == 3:
            pass

        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        actual_k = min(k, S.size(-1))

        U_k = U[..., :actual_k]
        S_k = S[..., :actual_k]
        Vh_k = Vh[..., :actual_k, :]

        x_reconstructed = U_k @ torch.diag_embed(S_k) @ Vh_k

        return x_reconstructed

    def forward(
            self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ):
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        tokens1, tokens2 = self.get_tokens(layer)
        B, Nf, C = feats.shape
        assert Nf == H * W, f"feats length {Nf} != H*W={H*W}"
        out1 = self.block1[layer](tokens1, feats, H, W)
        out2 = self.block2[layer](tokens2, feats, H, W)


        if not batch_first:
            out1 = out1.permute(1, 0, 2)
            out2 = out2.permute(1, 0, 2)

        out1 = self._apply_svd(out1, self.svd_k)
        out2 = self._apply_svd(out2, self.svd_k)

        if not batch_first:
            self.learnable_tokens1[layer] = out1.permute(1, 0, 2)
            self.learnable_tokens2[layer] = out2.permute(1, 0, 2)





class object_query_3D(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        init_param: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.init_param = init_param
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)

        nn.init.uniform_(self.learnable_tokens.data, -self.init_param, self.init_param)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)
        self.block = nn.ModuleList([
            Block(
                dim=512,
                num_heads=8,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0,
                attn_drop=0,
                drop_path=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                sr_ratio=1) for i in range(4)
        ])

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        B, Nf, C = feats.shape
        assert Nf == H * W, f"feats length {Nf} != H*W={H*W}"
        tokens = self.get_tokens(layer)
        self.learnable_tokens[layer]=self.block[id](tokens, feats, H, W)


