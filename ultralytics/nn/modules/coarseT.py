import copy
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from einops.einops import rearrange
from collections import OrderedDict
from .position_encoding import RoPEPositionEncodingSine
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

# 深度可分离卷积层
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

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
        self.aggregate = DepthwiseSeparableConv(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        # multi-head attention
        self.q_proj = DepthwiseSeparableConv(d_model, d_model, kernel_size=1)
        self.k_proj = DepthwiseSeparableConv(d_model, d_model, kernel_size=1)
        self.v_proj = DepthwiseSeparableConv(d_model, d_model, kernel_size=1) 
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = DepthwiseSeparableConv(d_model, d_model, kernel_size=1)

        # feed-forward network
        self.mlp = nn.Sequential(
            DepthwiseSeparableConv(d_model*2, d_model*2, kernel_size=1),
            nn.LeakyReLU(inplace = True),
            DepthwiseSeparableConv(d_model*2, d_model, kernel_size=1),
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
        query, key, value = self.q_proj(query.permute(0,3,1,2)).permute(0,2,3,1), self.k_proj(source.permute(0,3,1,2)).permute(0,2,3,1), self.v_proj(source.permute(0,3,1,2)).permute(0,2,3,1)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # 引入模态自适应融合系数
        alpha = torch.sigmoid(torch.mean(query - key, dim=[1, 2, 3], keepdim=True))
        query = alpha * query + (1 - alpha) * key

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim).permute(0,2,1).unsqueeze(-1)).squeeze(-1).permute(0,2,1) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # 获取x的尺寸
        batch_size, channels_x, height_x, width_x = x.size()
        # 调整m的尺寸以匹配x
        if m.size(2) != height_x or m.size(3) != width_x:
            m = F.interpolate(m, size=(height_x, width_x), mode='bilinear', align_corners=False)

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1)).permute(0, 2, 3, 1) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m

class VTFCF(nn.Module):
    """An Efficient Coarse Feature Transform module for RGB and infrared images."""

    def __init__(self, d_model):
        super(VTFCF, self).__init__()
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

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,x):
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