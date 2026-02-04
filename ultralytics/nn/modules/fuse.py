import math

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from matplotlib import pyplot as plt
from timm.models.layers import trunc_normal_
from torch.nn import Parameter

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        res= self.act(self.bn(self.conv(x)))
        return res

    def fuseforward(self, x):
        res = self.act(self.conv(x))

        return res
    
def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class MAM(nn.Module):
    def __init__(self, in_channel):
        super(MAM, self).__init__()
        # 特征降维路径：将拼接后的特征通道数逐步降低到64
        self.channel264 = nn.Sequential(
            Conv(in_channel*2, in_channel, 3, 1, 1),  # 拼接后通道数加倍，先降回原始通道数
            Conv(in_channel, in_channel//2, 3, 1, 1),  # 通道数减半
            Conv(in_channel//2, in_channel//4, 3, 1, 1),  # 通道数继续减半
            Conv(in_channel//4, 64, 1, 1, 0)  # 最终降为64通道
        )
        # 空间变换参数预测网络
        self.xy = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将特征图压缩为1x1
            nn.Conv2d(64, 2, 1, 1, 0)  # 预测2个空间变换参数（x和y方向的偏移）
        )
        # 初始化卷积层权重，使用较小的标准差，接近单位变换
        self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.xy[-1].bias.data.zero_()
        # 最终融合卷积层，将对齐后的特征与另一个模态的特征融合
        self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
    
    def forward(self, x):
        gr = x[0]  # 第一个模态的特征（可能是RGB）
        gt = x[1]  # 第二个模态的特征（可能是热红外）
        in_ = torch.cat([gr, gt], dim=1)  # 通道维度拼接
        
        # 特征降维处理
        n1 = self.channel264(in_)
        
        # 创建单位变换矩阵
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
        if in_.is_cuda:
            identity_theta = identity_theta.cuda().detach()
        
        # 预测空间变换参数
        shift_xy = self.xy(n1)
        bsize = shift_xy.shape[0]
        
        # 复制单位矩阵以匹配批次大小
        identity_theta = identity_theta.view(-1, 2, 3).repeat(bsize, 1, 1)
        
        # 根据预测的参数调整变换矩阵
        identity_theta[:, :, 2] += shift_xy.squeeze()
        
        # # 转换为半精度以匹配混合精度训练
        identity_theta = identity_theta.half()
        
        # 生成仿射变换网格
        wrap_grid = F.affine_grid(identity_theta.view(-1, 2, 3), in_.size(), align_corners=True).permute(0, 3, 1, 2)
        
        # 使用网格对第一个模态的特征进行重采样（对齐）
        wrap_gr = F.grid_sample(gr.float(), wrap_grid.float().permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 获取模型当前的默认数据类型
        default_dtype = next(self.fus1.parameters()).dtype

        # 将对齐后的特征与第二个模态的特征拼接并融合
        feature_fuse1 = self.fus1(torch.cat([wrap_gr.to(default_dtype), gt.to(default_dtype)], dim=1))
        return feature_fuse1

class MAM1(nn.Module):
    def __init__(self, in_channel):
        super(MAM1, self).__init__()
        self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
    def forward(self, x):
        gr = x[0]
        gt = x[1]
        feature_fuse1 = self.fus1(torch.cat([gr, gt], dim=1))
        return feature_fuse1
    
class MAM2(nn.Module):
    def __init__(self, in_channel):
        super(MAM2, self).__init__()
        self.channel264 = nn.Sequential(
            Conv(in_channel*2, 128, 3, 2, 1),
            Conv(128, 128, 1, 1, 1),
            convblock(128, 64, 3, 2, 1),
            convblock(64, 64, 1, 1, 0),
            convblock(64, 32, 3, 2, 1),
            convblock(32, 32, 1, 1, 0),
        )
        self.xy = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 2, 1, 1, 0)
        )
        self.scale1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 1, 1, 1, 0)
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 1, 1, 1, 0)
        )
        # Start with identity transformation
        self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.xy[-1].bias.data.zero_()
        self.scale1[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale1[-1].bias.data.zero_()
        self.scale2[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale2[-1].bias.data.zero_()
        self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
    def forward(self, x):
        gr = x[0]
        gt = x[1]
        in_ = torch.cat([gr, gt], dim=1)
        # in_ = gt -gr
        n1 = self.channel264(in_)
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
        if in_.is_cuda:
            identity_theta = identity_theta.cuda().detach()
        shift_xy = self.xy(n1)
        shift_s1 = self.scale1(n1)
        shift_s2 = self.scale2(n1)
        bsize = shift_xy.shape[0]
        identity_theta = identity_theta.view(-1, 2, 3).repeat(bsize, 1, 1)
        identity_theta[:, :, 2] += shift_xy.squeeze()
        identity_theta[:, :1, :1] += shift_s1.squeeze(2)
        identity_theta[:, 1, 1] += shift_s2.squeeze()
        identity_theta = identity_theta.half()
        wrap_grid = F.affine_grid(identity_theta.view(-1, 2, 3), in_.size(), align_corners=True).permute(0, 3, 1,2)

        wrap_gr = F.grid_sample(gr.float(), wrap_grid.float().permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros',align_corners=True)
        
        # 获取模型当前的默认数据类型
        default_dtype = next(self.fus1.parameters()).dtype

        # 将对齐后的特征与第二个模态的特征拼接并融合
        fuse = self.fus1(torch.cat([wrap_gr.to(default_dtype), gt.to(default_dtype)], dim=1))

        return fuse

# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        # x = torch.cat((x1, x2), dim=1)
        # avg1 = self.avg_pool(x1).view(B, self.dim)
        avg1 = torch.mean(x1, dim=[2, 3], keepdim=True).view(B, self.dim)
        avg2 = torch.mean(x2, dim=[2, 3], keepdim=True).view(B, self.dim)
        # avg2 = self.avg_pool(x2).view(B, self.dim)
        max1 = self.max_pool(x1).view(B, self.dim)
        max2 = self.max_pool(x2).view(B, self.dim)
        avg = avg1+avg2
        max = max1+max2
        y = torch.cat((max, avg), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights
    
class FRM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FRM, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        # out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        x1 = x1 + self.lambda_c * channel_weights[0] * x1
        # out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        x2 = x2 + self.lambda_c * channel_weights[1] * x2
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_s * spatial_weights[0] * x1
        out_x2 = x2 + self.lambda_s * spatial_weights[1] * x2
        out = out_x1 + out_x2
        return out

class DynamicConvFusion(nn.Module):
    def __init__(self, dim, reduction=4, lambda_c=0.5, lambda_s=0.5):
        super(DynamicConvFusion, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        
        # 特征提取模块
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # 动态卷积参数生成器
        self.conv_weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//reduction, 9*dim, kernel_size=1, bias=False)  # 3x3卷积核参数
        )
        
        # 注意力权重生成器
        self.attention_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//reduction, 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x1, x2):
        # 拼接特征
        concat_features = torch.cat([x1, x2], dim=1)
        shared_features = self.feature_extractor(concat_features)
        
        # 生成动态卷积权重
        conv_weights = self.conv_weight_generator(shared_features)
        B, C, H, W = x1.size()
        conv_weights = conv_weights.view(B, C, 9, 1, 1)  # 重塑为[B, C, 9, 1, 1]
        
        # 生成注意力权重
        attn_weights = self.attention_generator(shared_features)
        attn_x1, attn_x2 = attn_weights[:, 0:1, :, :], attn_weights[:, 1:2, :, :]
        
        # 应用动态卷积
        x1_reshaped = x1.reshape(1, B*C, H, W)
        x2_reshaped = x2.reshape(1, B*C, H, W)
        
        # 使用分组卷积实现动态卷积
        dynamic_x1 = F.conv2d(
            x1_reshaped, 
            weight=conv_weights.reshape(B*C, 1, 3, 3), 
            bias=None, 
            stride=1, 
            padding=1, 
            groups=B*C
        ).view(B, C, H, W)
        
        dynamic_x2 = F.conv2d(
            x2_reshaped, 
            weight=conv_weights.reshape(B*C, 1, 3, 3), 
            bias=None, 
            stride=1, 
            padding=1, 
            groups=B*C
        ).view(B, C, H, W)
        
        # 加权融合
        out_x1 = x1 + self.lambda_c * attn_x1 * dynamic_x1
        out_x2 = x2 + self.lambda_c * attn_x2 * dynamic_x2
        
        # 最终融合
        out = self.lambda_s * out_x1 + (1-self.lambda_s) * out_x2
        return out

from torchvision.ops import deform_conv2d
class DeformableFRM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=0.5):
        super(DeformableFRM, self).__init__()
        self.lambda_c = lambda_c
        
        # 通道注意力模块保持不变
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        
        # 可变形卷积参数
        self.deform_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.offset_conv = nn.Conv2d(dim * 2, 2 * 3 * 3, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(dim * 2, 1 * 3 * 3, kernel_size=3, padding=1)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 偏移卷积初始化为0，从零开始学习
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        # 掩码卷积初始化为0.5
        nn.init.constant_(self.mask_conv.weight, 0)
        nn.init.constant_(self.mask_conv.bias, 0.5)
        
        # 可变形卷积正常初始化
        nn.init.kaiming_normal_(self.deform_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.deform_conv.bias is not None:
            nn.init.constant_(self.deform_conv.bias, 0)

    def forward(self, x1, x2):
        # 通道注意力
        channel_weights = self.channel_weights(x1, x2)
        x1_c = x1 + self.lambda_c * channel_weights[0] * x1
        x2_c = x2 + self.lambda_c * channel_weights[1] * x2
        
        # 拼接特征用于生成偏移和掩码
        concat_features = torch.cat([x1_c, x2_c], dim=1)
        
        # 生成偏移量 (batch_size, 2*3*3, H, W)
        offset = self.offset_conv(concat_features)
        
        # 生成掩码 (batch_size, 1*3*3, H, W)
        mask = torch.sigmoid(self.mask_conv(concat_features))
        
        # 应用可变形卷积融合特征
        fused_features = deform_conv2d(
            x1_c, 
            offset, 
            self.deform_conv.weight, 
            self.deform_conv.bias,
            padding=1,
            mask=mask
        )
        
        # 最终融合
        out = fused_features + x2_c
        return out

class Fuse1(nn.Module):
    def __init__(self, in_channel):
        super(Fuse1, self).__init__()
        # self.esam = ESAM(in_channel)
        # self.DSMM = DSMM1(in_channel)
        self.mam = MAM2(in_channel)
        self.fuse = FRM(in_channel)
        # self.fuse = DynamicConvFusion(in_channel)
        # self.fuse = OptimizedDeformableConvFusion(in_channel)
        # self.dy = DynamicConv1(in_channel, in_channel, 3, 1, 1)
        # self.fus1 = Conv(in_channel * 2, in_channel, 1, 1, 0)
    def forward(self,x):
        rgb = x[0]
        t = x[1]
        # t1 = self.esam(t)
        # rgb1 = self.DSMM(rgb,t)
        # x = [rgb, t]
        x = [t, rgb]
        # x = [t, rgb]
        gr = self.mam(x)
        # final = gr+t
        # map_rgb = torch.unsqueeze(torch.mean(final, 1), 1)
        # score2 = F.interpolate(map_rgb, size=(128, 128), mode="bilinear", align_corners=True)
        # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
        # depth = (score2 - score2.min()) / (score2.max() - score2.min())
        # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        # plt.imshow(feature_img)
        # plt.show()
        # plt.savefig("2.png")
        # gt = self.mam(x)
        # dy_rgb = self.dy(gr,t)
        final = self.fuse(gr, t)##
        # final = gr + t
        # final = self.fuse(rgb, gt)
        # fuse = self.fus1(torch.cat([gr, t], dim=1))
        # final = gr+t
        # map_rgb = torch.unsqueeze(torch.mean(final, 1), 1)
        # score2 = F.interpolate(map_rgb, size=(80, 80), mode="bilinear", align_corners=True)
        # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
        # depth = (score2 - score2.min()) / (score2.max() - score2.min())
        # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        # plt.imshow(feature_img)
        # plt.show()
        # plt.savefig("1.png")
        return final

class Fuse2(nn.Module):
    def __init__(self, in_channel):
        super(Fuse2, self).__init__()
        self.mam = MAM2(in_channel)
        self.fuse = DynamicConvFusion(in_channel)

    def forward(self,x):
        rgb = x[0]
        t = x[1]
        x = [t, rgb]
        gr = self.mam(x)
        final = self.fuse(gr, t)
        return final
    
class Fuse3(nn.Module):
    def __init__(self, in_channel):
        super(Fuse3, self).__init__()
        self.mam = MAM2(in_channel)
        self.fuse = DeformableFRM(in_channel)

    def forward(self,x):
        rgb = x[0]
        t = x[1]
        x = [t, rgb]
        gr = self.mam(x)
        final = self.fuse(gr, t)
        return final
    
class Fuse4(nn.Module):
    def __init__(self, in_channel):
        super(Fuse4, self).__init__()
        self.mam = MAM2(in_channel)
        # self.fuse = DynamicConvFusion(in_channel)

    def forward(self,x):
        rgb = x[0]
        t = x[1]
        x = [t, rgb]
        gr = self.mam(x)
        final = gr + t
        return final
    
