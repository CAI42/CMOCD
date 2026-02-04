import torch
import torch.nn as nn
from torch.nn import init, Sequential
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        # inner_dim = 682
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # b, 65, 1024, heads = 8
        b, n, _, h = *x.shape, self.heads

        # self.to_qkv(x): b, 65, 64*8*3
        # qkv: b, 65, 64*8
        # x = x.unsqueeze(dim=2)       682

        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim=-1)

        # b, 65, 64, 8
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q_s, q_t = torch.chunk(q, 2, 2)
        k_s, k_t = torch.chunk(k, 2, 2)
        v_s, v_t = torch.chunk(v, 2, 2)
        #
        # dots:b, 65, 64, 64
        # dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # mask_value = -torch.finfo(dots.dtype).max
        # dots_t = torch.einsum('bhid,bhjd->bhij', q_t, k_s) * self.scale
        dots_t = torch.einsum('bhid,bhjd->bhij', q_t, k_t) * self.scale
        dots_s = torch.einsum('bhid,bhjd->bhij', q_s, k_t) * self.scale
        # dots_s = torch.einsum('bhid,bhjd->bhij', q_s, k_s) * self.scale
        # mask_value = -torch.finfo(dots.dtype).max
        # if mask is not None:
        #     mask = F.pad(mask.flatten(1), (1, 0), value=True)
        #     assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = mask[:, None, :] * mask[:, :, None]
        #     dots.masked_fill_(~mask, mask_value)
        #     del mask
        #
        # attn:b, 65, 64, 64
        # attn = dots.softmax(dim=-1)
        attn_s = dots_s.softmax(dim=-1)
        attn_t = dots_t.softmax(dim=-1)

        # 使用einsum表示矩阵乘法：
        # out:b, 65, 64, 8
        # out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out_s = torch.einsum('bhij,bhjd->bhid', attn_s, v_s)
        out_t = torch.einsum('bhij,bhjd->bhid', attn_t, v_t)
        out = torch.cat([out_s, out_t], dim=2)
        # out:b, 64, 65*8
        out = rearrange(out, 'b h n d -> b n (h d)')

        # out:b, 64, 1024
        out = self.to_out(out)
        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout): #1,1,1,32,16,0.1
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
    
class CEM1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.transformer = Transformer(dim=1, depth=1, heads=1, dim_head=16, mlp_dim=8, dropout=0.1)
        self.attention_weight = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.pos_embedding = nn.Parameter(torch.randn(1, channels*2, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, rgb,t):
        x = torch.cat([rgb,t], dim=1)
        b, c, h, w = rgb.size()  # 32, 256, 72, 36
        input = self.gap(x).squeeze(-1)  # 32， 256， 72*36=2592
        _, c, _ = input.shape
        input = input + self.pos_embedding[:, :(c)]
        input = self.dropout(input)
        output = self.transformer(input)  # 32, 256, 1
        output = torch.unsqueeze(output, dim=3)  # 32, 256, 1, 1
        weight = torch.sigmoid(output)  # 32, 256, 1, 1
        final = (weight * x).view(b,2,c//2,h,w)
        rgb_ = final[:,0,:,:,:]
        t_ = final[:,1,:,:,:]
        # fuse = rgb_+t_
        return rgb_, t_
    
class CEM2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.transformer = Transformer(dim=1, depth=1, heads=1, dim_head=16, mlp_dim=8, dropout=0.1)
        self.attention_weight = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.pos_embedding = nn.Parameter(torch.randn(1, channels*2, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, rgb, t):
        # 确保两个输入的通道数相同
        assert rgb.size(1) == t.size(1), "RGB and thermal images must have the same number of channels"
        
        x = torch.cat([rgb, t], dim=1)
        b, c, h, w = rgb.size()  # 32, 256, 72, 36
        
        # 全局平均池化，得到(b, c*2, 1, 1)的张量
        input = self.gap(x)  # (b, c*2, 1, 1)
        input = input.squeeze(-1).squeeze(-1)  # (b, c*2, 1)
        input = input.permute(0, 2, 1)  # 调整为Transformer期望的形状 (b, 1, c*2)
        
        # 添加位置编码
        input = input + self.pos_embedding[:, :input.size(-1)].permute(0, 2, 1)
        input = self.dropout(input)
        
        # 通过transformer生成注意力权重
        output = self.transformer(input)  # (b, 1, c*2)
        output = output.permute(0, 2, 1).unsqueeze(-1)  # (b, c*2, 1, 1)
        
        # 生成sigmoid激活的注意力权重
        weight = torch.sigmoid(output)  # (b, c*2, 1, 1)
        
        # 应用注意力权重
        weighted_features = weight * x  # (b, c*2, h, w)
        
        # 分离RGB和热成像特征
        rgb_channels = t_channels = weighted_features.size(1) // 2
        rgb_ = weighted_features[:, :rgb_channels, :, :]
        t_ = weighted_features[:, rgb_channels:, :, :]
        
        return rgb_, t_

class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop_t = nn.Dropout(attn_pdrop)
        self.attn_drop_s = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        q_s, q_t = torch.chunk(q,2,2)
        k_s, k_t = torch.chunk(k,2,3)
        v_s, v_t = torch.chunk(v,2,2)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        # att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att_s = torch.matmul(q_s, k_t) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        # att_s = torch.matmul(q_s, k_s) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att_t = torch.matmul(q_t, k_t) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        # att_t = torch.matmul(q_t, k_s) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        # if attention_weights is not None:
        #     att = att * attention_weights
        # if attention_mask is not None:
        #     att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att_s = torch.softmax(att_s, -1)
        att_s = self.attn_drop_s(att_s)

        att_t = torch.softmax(att_t, -1)
        att_t = self.attn_drop_t(att_t)

        # output
        # out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_t = torch.matmul(att_t, v_t).permute(0, 2, 1, 3).contiguous().view(b_s, nq//2, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_s = torch.matmul(att_s, v_s).permute(0, 2, 1, 3).contiguous().view(b_s, nq//2, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out1 = torch.cat([out_s, out_t], dim=1)
        out = self.resid_drop(self.out_proj(out1))  # (b_s, nq, d_model)
        return out
    
class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model,d_k,d_v,h)
        # self.sa = Attention(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()
        # x = x + self.sa(self.ln_input(x))
        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))
        return x


class GPT1(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=4, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)
        self.cem = CEM1(d_model)
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)
        """
        rgb = x[0]
        ir = x[1]
        assert rgb.shape[0] == ir.shape[0]
        bs, c, h, w = rgb.shape
        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb)
        ir_fea = self.avgpool(ir)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')
        rgb,ir = self.cem(rgb,ir)
        rgb_fea_out+=rgb
        ir_fea_out+=ir
        # out = rgb_fea_out+ir_fea_out
        # map_rgb = torch.unsqueeze(torch.mean(out, 1), 1)
        # score2 = F.interpolate(map_rgb, size=(256, 256), mode="bilinear", align_corners=True)
        # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
        # depth = (score2 - score2.min()) / (score2.max() - score2.min())
        # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        # plt.imshow(feature_img)
        # plt.show()
        # plt.savefig("1.png")
        return rgb_fea_out, ir_fea_out