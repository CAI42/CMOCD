import copy
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from einops.einops import rearrange
from collections import OrderedDict
from .position_encoding import RoPEPositionEncodingSine
# from .GPT import Attention
import numpy as np
from einops.einops import rearrange, repeat

if hasattr(F, 'scaled_dot_product_attention'):
    FLASH_AVAILABLE = True
    from torch.backends.cuda import sdp_kernel
else:
    FLASH_AVAILABLE = False

def crop_feature(query, key, value, x_mask, source_mask):
    mask_h0, mask_w0, mask_h1, mask_w1 = x_mask[0].sum(-2)[0], x_mask[0].sum(-1)[0], source_mask[0].sum(-2)[0], source_mask[0].sum(-1)[0]
    query = query[:, :mask_h0, :mask_w0, :]
    key = key[:, :mask_h1, :mask_w1, :]
    value = value[:, :mask_h1, :mask_w1, :]
    return query, key, value, mask_h0, mask_w0

def pad_feature(m, mask_h0, mask_w0, x_mask):
    bs, L, H, D = m.size()
    m = m.view(bs, mask_h0, mask_w0, H, D)
    if mask_h0 != x_mask.size(-2):
        m = torch.cat([m, torch.zeros(m.size(0), x_mask.size(-2)-mask_h0, x_mask.size(-1), H, D, device=m.device, dtype=m.dtype)], dim=1)
    elif mask_w0 != x_mask.size(-1):
        m = torch.cat([m, torch.zeros(m.size(0), x_mask.size(-2), x_mask.size(-1)-mask_w0, H, D, device=m.device, dtype=m.dtype)], dim=2)
    return m

class Attention(Module):
    def __init__(self, no_flash=False, nhead=8, dim=256, fp32=False):
        super().__init__()
        self.flash = FLASH_AVAILABLE and not no_flash
        self.nhead = nhead
        self.dim = dim
        self.fp32 = fp32
        
    def attention(self, query, key, value, q_mask=None, kv_mask=None):
        assert q_mask is None and kv_mask is None, "Not support generalized attention mask yet."
        if self.flash and not self.fp32:
            args = [x.contiguous() for x in [query, key, value]]
            with sdp_kernel(enable_math= False, enable_flash= True, enable_mem_efficient= False):
                out = F.scaled_dot_product_attention(*args)
        elif self.flash:
            args = [x.contiguous() for x in [query, key, value]]
            out = F.scaled_dot_product_attention(*args)
        else:
            QK = torch.einsum("nlhd,nshd->nlsh", query, key)
    
            # Compute the attention and the weighted average
            softmax_temp = 1. / query.size(3)**.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)

            out = torch.einsum("nlsh,nshd->nlhd", A, value)
        return out

    def _forward(self, query, key, value, q_mask=None, kv_mask=None):
        if q_mask is not None:
            query, key, value, mask_h0, mask_w0 = crop_feature(query, key, value, q_mask, kv_mask)

        if self.flash:
            query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim), [query, key, value])
        else:
            query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim), [query, key, value])

        m = self.attention(query, key, value, q_mask=None, kv_mask=None)

        if self.flash:
            m = rearrange(m, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

        if q_mask is not None:
            m = pad_feature(m, mask_h0, mask_w0, q_mask)
        
        return m
    
    def forward(self, query, key, value, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            if FLASH_AVAILABLE: # pytorch scaled_dot_product_attention
                queries: [N, H, L, D]
                keys: [N, H, S, D]
                values: [N, H, S, D]
            else:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        bs = query.size(0)
        if bs == 1 or q_mask is None:            
            m = self._forward(query, key, value, q_mask=q_mask, kv_mask=kv_mask)
        else: # for faster trainning with padding mask while batch size > 1
            m_list = []
            for i in range(bs):
                m_list.append(self._forward(query[i:i+1], key[i:i+1], value[i:i+1], q_mask=q_mask[i:i+1], kv_mask=kv_mask[i:i+1]))
            m = torch.cat(m_list, dim=0)
        return m


class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size0=4,
                 agg_size1=4,
                 no_flash=False,
                 rope=False,
                 npe=None,
                 fp32=False,
                 ):
        super(AG_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False, groups=d_model) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False) 
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        # self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature
        query, source = self.norm1(self.aggregate(x).permute(0,2,3,1)), self.norm1(self.max_pool(source).permute(0,2,3,1)) # [N, H, W, C]
        if x_mask is not None:
            x_mask, source_mask = map(lambda x: self.max_pool(x.float()).bool(), [x_mask, source_mask])
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # print("x:",x.shape,"m:",m.shape)#额滴神啊，为什么这里尺寸数不一样了
        # 获取x的尺寸
        batch_size, channels_x, height_x, width_x = x.size()
        # 调整m的尺寸以匹配x
        if m.size(2) != height_x or m.size(3) != width_x:
            m = F.interpolate(m, size=(height_x, width_x), mode='bilinear', align_corners=False)

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m

class EfficientCoarseFeatureTransform(nn.Module):
    """An Efficient Coarse Feature Transform module for RGB and infrared images."""

    def __init__(self, d_model):
        super(EfficientCoarseFeatureTransform, self).__init__()
        self.fp32 = True
        self.d_model = d_model
        self.nhead = 8
        self.no_flash = False
        self.layer_names = ['self', 'cross'] * 4
        self.agg_size0, self.agg_size1 = 4,4
        self.npe= [832, 832, 832, 832]
        self.rope = True

        self_layer = AG_RoPE_EncoderLayer(self.d_model, self.nhead, self.agg_size0, self.agg_size1,
                                          self.no_flash, self.rope,self.npe, self.fp32)
        cross_layer = AG_RoPE_EncoderLayer(self.d_model, self.nhead, self.agg_size0, self.agg_size1, 
                                           self.no_flash, False, self.npe, self.fp32)
        self.layers = nn.ModuleList([copy.deepcopy(self_layer) if _ == 'self' else copy.deepcopy(cross_layer) for _ in self.layer_names])
        self._reset_parameters()
        # self.FeatureWeightedFusion = FeatureWeightedFusion()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,x):
    # def forward(self, rgb_feat, ir_feat, rgb_mask=None, ir_mask=None):
        """
        Args:
            rgb_feat (torch.Tensor): [N, C, H, W] RGB image features
            ir_feat (torch.Tensor): [N, C, H, W] Infrared image features
            rgb_mask (torch.Tensor): [N, L] (optional)
            ir_mask (torch.Tensor): [N, S] (optional)
        """
        rgb_feat = x[0]
        ir_feat = x[1]
        assert rgb_feat.shape[0] == ir_feat.shape[0]
        H0, W0, H1, W1 = rgb_feat.size(-2), rgb_feat.size(-1), ir_feat.size(-2), ir_feat.size(-1)
        bs = rgb_feat.shape[0]

        rgb_mask = None
        ir_mask = None
        
        feature_cropped = False
        if bs == 1 and rgb_mask is not None and ir_mask is not None:
            mask_H0, mask_W0, mask_H1, mask_W1 = rgb_mask.size(-2), rgb_mask.size(-1), ir_mask.size(-2), ir_mask.size(-1)
            mask_h0, mask_w0, mask_h1, mask_w1 = rgb_mask[0].sum(-2)[0], rgb_mask[0].sum(-1)[0], ir_mask[0].sum(-2)[0], ir_mask[0].sum(-1)[0]
            mask_h0, mask_w0, mask_h1, mask_w1 = mask_h0//self.agg_size0*self.agg_size0, mask_w0//self.agg_size0*self.agg_size0, mask_h1//self.agg_size1*self.agg_size1, mask_w1//self.agg_size1*self.agg_size1
            rgb_feat = rgb_feat[:, :, :mask_h0, :mask_w0]
            ir_feat = ir_feat[:, :, :mask_h1, :mask_w1]
            feature_cropped = True

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            # print('i:', i, 'name:', name, 'rgb_feat:', rgb_feat.shape, 'ir_feat:', ir_feat.shape)
            if feature_cropped:
                rgb_mask, ir_mask = None, None
            if name == 'self':
                rgb_feat = layer(rgb_feat, rgb_feat, rgb_mask, rgb_mask)
                ir_feat = layer(ir_feat, ir_feat, ir_mask, ir_mask)
            elif name == 'cross':
                rgb_feat = layer(rgb_feat, ir_feat, rgb_mask, ir_mask)
                ir_feat = layer(ir_feat, rgb_feat, ir_mask, rgb_mask)                
            else:
                raise KeyError
        # print('feature_cropped:', feature_cropped)
        # if feature_cropped:
        #     # padding feature
        #     bs, c, mask_h0, mask_w0 = rgb_feat.size()
        #     if mask_h0 != mask_H0:
        #         rgb_feat = torch.cat([rgb_feat, torch.zeros(bs, c, mask_H0-mask_h0, mask_W0, device=rgb_feat.device, dtype=rgb_feat.dtype)], dim=-2)
        #     elif mask_w0 != mask_W0:
        #         rgb_feat = torch.cat([rgb_feat, torch.zeros(bs, c, mask_H0, mask_W0-mask_w0, device=rgb_feat.device, dtype=rgb_feat.dtype)], dim=-1)

        #     bs, c, mask_h1, mask_w1 = ir_feat.size()
        #     if mask_h1 != mask_H1:
        #         ir_feat = torch.cat([ir_feat, torch.zeros(bs, c, mask_H1-mask_h1, mask_W1, device=ir_feat.device, dtype=ir_feat.dtype)], dim=-2)
        #     elif mask_w1 != mask_W1:
        #         ir_feat = torch.cat([ir_feat, torch.zeros(bs, c, mask_H1, mask_W1-mask_w1, device=ir_feat.device, dtype=ir_feat.dtype)], dim=-1)
        #     print('rgb_feat:', rgb_feat.shape, 'ir_feat:', ir_feat.shape)
        #     print('mask_H0, mask_W0:', mask_H0, mask_W0, 'mask_H1, mask_W1:', mask_H1, mask_W1)
        #     print('mask_h0, mask_w0:', mask_h0, mask_w0, 'mask_h1, mask_w1:', mask_h1, mask_w1)
        # rgb_feat, ir_feat =self.FeatureWeightedFusion(rgb_feat, ir_feat)
        return rgb_feat, ir_feat
    

# from sklearn.metrics import mutual_info_score
# class FeatureWeightedFusion(nn.Module):
#     def __init__(self):
#         super(FeatureWeightedFusion, self).__init__()

#     def compute_mutual_info(self, feat1, feat2):
#         # 将特征展平为一维向量
#         feat1_flat = feat1.view(-1).cpu().numpy()
#         feat2_flat = feat2.view(-1).cpu().numpy()
#         # 计算互信息
#         mi = mutual_info_score(feat1_flat, feat2_flat)
#         return mi

#     def forward(self, visible_feat, infrared_feat):
#         # 计算互信息
#         mi = self.compute_mutual_info(visible_feat, infrared_feat)
#         # 简单示例：根据互信息计算动态权重
#         weight_visible = mi / (mi + 1e-8)
#         weight_infrared = 1 - weight_visible
#         # 特征加权
#         weighted_visible = visible_feat * weight_visible
#         weighted_infrared = infrared_feat * weight_infrared
#         # # 特征拼接
#         # fused_feat = torch.cat([weighted_visible, weighted_infrared], dim=1)
#         return weighted_visible, weighted_infrared


class Adaptive_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size0=4,
                 agg_size1=4,
                 no_flash=False,
                 rope=False,
                 npe=None,
                 fp32=False,
                 ):
        super(Adaptive_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False, groups=d_model) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False) 
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        # self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature
        query, source = self.norm1(self.aggregate(x).permute(0,2,3,1)), self.norm1(self.max_pool(source).permute(0,2,3,1)) # [N, H, W, C]
        if x_mask is not None:
            x_mask, source_mask = map(lambda x: self.max_pool(x.float()).bool(), [x_mask, source_mask])
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # 引入模态自适应融合系数
        alpha = torch.sigmoid(torch.mean(query - key, dim=[1, 2, 3], keepdim=True))
        query = alpha * query + (1 - alpha) * key

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # print("x:",x.shape,"m:",m.shape)#额滴神啊，为什么这里尺寸数不一样了
        # 获取x的尺寸
        batch_size, channels_x, height_x, width_x = x.size()
        # 调整m的尺寸以匹配x
        if m.size(2) != height_x or m.size(3) != width_x:
            m = F.interpolate(m, size=(height_x, width_x), mode='bilinear', align_corners=False)

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m

class AdaptiveCoarseFeatureTransform(nn.Module):
    """An Adaptive Coarse Feature Transform module for RGB and infrared images."""

    def __init__(self, d_model):
        super(AdaptiveCoarseFeatureTransform, self).__init__()
        self.fp32 = True
        self.d_model = d_model
        self.nhead = 8
        self.no_flash = False
        self.layer_names = ['self', 'cross'] * 4
        self.agg_size0, self.agg_size1 = 4,4
        self.npe= [832, 832, 832, 832]
        self.rope = True

        self_layer = Adaptive_RoPE_EncoderLayer(self.d_model, self.nhead, self.agg_size0, self.agg_size1,
                                          self.no_flash, self.rope,self.npe, self.fp32)
        cross_layer = Adaptive_RoPE_EncoderLayer(self.d_model, self.nhead, self.agg_size0, self.agg_size1, 
                                           self.no_flash, False, self.npe, self.fp32)
        self.layers = nn.ModuleList([copy.deepcopy(self_layer) if _ == 'self' else copy.deepcopy(cross_layer) for _ in self.layer_names])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,x):
        """
        Args:
            rgb_feat (torch.Tensor): [N, C, H, W] RGB image features
            ir_feat (torch.Tensor): [N, C, H, W] Infrared image features
            rgb_mask (torch.Tensor): [N, L] (optional)
            ir_mask (torch.Tensor): [N, S] (optional)
        """
        rgb_feat = x[0]
        ir_feat = x[1]
        assert rgb_feat.shape[0] == ir_feat.shape[0]
        H0, W0, H1, W1 = rgb_feat.size(-2), rgb_feat.size(-1), ir_feat.size(-2), ir_feat.size(-1)
        bs = rgb_feat.shape[0]

        rgb_mask = None
        ir_mask = None
        
        feature_cropped = False
        if bs == 1 and rgb_mask is not None and ir_mask is not None:
            mask_H0, mask_W0, mask_H1, mask_W1 = rgb_mask.size(-2), rgb_mask.size(-1), ir_mask.size(-2), ir_mask.size(-1)
            mask_h0, mask_w0, mask_h1, mask_w1 = rgb_mask[0].sum(-2)[0], rgb_mask[0].sum(-1)[0], ir_mask[0].sum(-2)[0], ir_mask[0].sum(-1)[0]
            mask_h0, mask_w0, mask_h1, mask_w1 = mask_h0//self.agg_size0*self.agg_size0, mask_w0//self.agg_size0*self.agg_size0, mask_h1//self.agg_size1*self.agg_size1, mask_w1//self.agg_size1*self.agg_size1
            rgb_feat = rgb_feat[:, :, :mask_h0, :mask_w0]
            ir_feat = ir_feat[:, :, :mask_h1, :mask_w1]
            feature_cropped = True

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            # print('i:', i, 'name:', name, 'rgb_feat:', rgb_feat.shape, 'ir_feat:', ir_feat.shape)
            if feature_cropped:
                rgb_mask, ir_mask = None, None
            if name == 'self':
                rgb_feat = layer(rgb_feat, rgb_feat, rgb_mask, rgb_mask)
                ir_feat = layer(ir_feat, ir_feat, ir_mask, ir_mask)
            elif name == 'cross':
                rgb_feat = layer(rgb_feat, ir_feat, rgb_mask, ir_mask)
                ir_feat = layer(ir_feat, rgb_feat, ir_mask, rgb_mask)                
            else:
                raise KeyError
        return rgb_feat, ir_feat

#粗配准
class CoarseMatching(nn.Module):
    def __init__(self, ddd):
        super().__init__()
        self.d_model=ddd
        self.data ={
                    'hw0_i': [640, 640],
                    'hw1_i': [640, 640],
                    'hw0_c': [8, 8],
                    'hw1_c': [8, 8],
                    'hw0_f': [64, 64],
                    'hw1_f': [64, 64],
                    'bs': 1,
                    'spv_b_ids': [0],#介是嘛
                    'spv_i_ids': [0],
                    'spv_j_ids': [0],
                    
                    }
        self.thr = 0.2
        self.border_rm = 2
        self.temperature = 0.1
        self.skip_softmax = False # False for full model and True for optimized model
        self.fp16matmul = False # False for full model and True for optimized model
        self.train_coarse_percent =  1#0.2  # training tricks: save GPU memory
        self.train_pad_num_gt_min = 20#200  # training tricks: avoid DDP deadlock

    def forward(self, x, mask_c0=None, mask_c1=None):
        feat_c0 = x[0][0]
        feat_c1 = x[0][1]
        assert feat_c0.shape[0] == feat_c1.shape[0]
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)
        # print('N, L, S, C:',N, L, S, C)
        # 获取批次大小和通道数
        batch_size, channels, height, width = feat_c0.size()  
        print('batch_size, channels, height, width:',batch_size, channels, height, width)      
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])
        # 将空间维度展平为序列维度
        feat_c0_flat = feat_c0.view(batch_size, channels, height * width).permute(0, 2, 1)  # [N, L, C]
        feat_c1_flat = feat_c1.view(batch_size, channels, height * width).permute(0, 2, 1)  # [N, S, C]

        data=self.data
        if self.fp16matmul:
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            if mask_c0 is not None:
                sim_matrix = sim_matrix.masked_fill(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e4)
        else:
            with torch.autocast(enabled=False, device_type='cuda'):
                sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0_flat, feat_c1_flat) / self.temperature
                if mask_c0 is not None:
                    sim_matrix = sim_matrix.float().masked_fill(
                        ~(mask_c0[..., None] * mask_c1[:, None]).bool(), -1e9)

        if self.skip_softmax:
            sim_matrix = sim_matrix
        else:
            sim_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data.update({'conf_matrix': sim_matrix})
        data.update(**self.get_coarse_match(sim_matrix, data))
        return data

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        mask = conf_matrix > self.thr
        # print('mask:',mask.size())
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes_lengths)
        if 'mask0' not in data:
            self.mask_border(mask, self.border_rm, False)
        else:
            self.mask_border_with_padding(mask, self.border_rm, False, data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)', **axes_lengths)

        mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        if self.training:
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = self.compute_max_candidates(data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            print('self.train_pad_num_gt_min',self.train_pad_num_gt_min,'num_matches_train',num_matches_train)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(num_matches_pred, (num_matches_train - self.train_pad_num_gt_min,), device=_device)

            gt_pad_indices = torch.randint(len(data['spv_b_ids']), (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),), device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']], [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack([i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]], dim=1) * scale1
        m_bids = b_ids[mconf != 0]
        coarse_matches.update({
            'm_bids': m_bids,
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })
        return coarse_matches

    def mask_border(self, m, b, v):
        if b <= 0:
            return
        m[:, :b] = v
        m[:, :, :b] = v
        m[:, :, :, :b] = v
        m[:, :, :, :, :b] = v
        m[:, -b:] = v
        m[:, :, -b:] = v
        m[:, :, :, -b:] = v
        m[:, :, :, :, -b:] = v

    def mask_border_with_padding(self, m, bd, v, p_m0, p_m1):
        if bd <= 0:
            return
        m[:, :bd] = v
        m[:, :, :bd] = v
        m[:, :, :, :bd] = v
        m[:, :, :, :, :bd] = v
        h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
        h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
        for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
            m[b_idx, h0 - bd:] = v
            m[b_idx, :, w0 - bd:] = v
            m[b_idx, :, :, h1 - bd:] = v
            m[b_idx, :, :, :, w1 - bd:] = v

    def compute_max_candidates(self, p_m0, p_m1):
        h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
        h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
        max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
        return max_cand

