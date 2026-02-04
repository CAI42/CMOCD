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

# class MAM(nn.Module):
#     def __init__(self, in_channel):
#         super(MAM, self).__init__()
#         # 特征降维路径：将拼接后的特征通道数逐步降低到64
#         self.channel264 = nn.Sequential(
#             Conv(in_channel*2, in_channel, 3, 1, 1),  # 拼接后通道数加倍，先降回原始通道数
#             Conv(in_channel, in_channel//2, 3, 1, 1),  # 通道数减半
#             Conv(in_channel//2, in_channel//4, 3, 1, 1),  # 通道数继续减半
#             Conv(in_channel//4, 64, 1, 1, 0)  # 最终降为64通道
#         )
#         # 空间变换参数预测网络
#         self.xy = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将特征图压缩为1x1
#             nn.Conv2d(64, 2, 1, 1, 0)  # 预测2个空间变换参数（x和y方向的偏移）
#         )
#         # 初始化卷积层权重，使用较小的标准差，接近单位变换
#         self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
#         self.xy[-1].bias.data.zero_()
#         # 最终融合卷积层，将对齐后的特征与另一个模态的特征融合
#         self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
    
#     def forward(self, x):
#         gr = x[0]  # 第一个模态的特征（可能是RGB）
#         gt = x[1]  # 第二个模态的特征（可能是热红外）
#         in_ = torch.cat([gr, gt], dim=1)  # 通道维度拼接
        
#         # 特征降维处理
#         n1 = self.channel264(in_)
        
#         # 创建单位变换矩阵
#         identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
#         if in_.is_cuda:
#             identity_theta = identity_theta.cuda().detach()
        
#         # 预测空间变换参数
#         shift_xy = self.xy(n1)
#         bsize = shift_xy.shape[0]
        
#         # 复制单位矩阵以匹配批次大小
#         identity_theta = identity_theta.view(-1, 2, 3).repeat(bsize, 1, 1)
        
#         # 根据预测的参数调整变换矩阵
#         identity_theta[:, :, 2] += shift_xy.squeeze()
        
#         # # 转换为半精度以匹配混合精度训练
#         identity_theta = identity_theta.half()
        
#         # 生成仿射变换网格
#         wrap_grid = F.affine_grid(identity_theta.view(-1, 2, 3), in_.size(), align_corners=True).permute(0, 3, 1, 2)
        
#         # 使用网格对第一个模态的特征进行重采样（对齐）
#         wrap_gr = F.grid_sample(gr.float(), wrap_grid.float().permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
        
#         # 获取模型当前的默认数据类型
#         default_dtype = next(self.fus1.parameters()).dtype

#         # 将对齐后的特征与第二个模态的特征拼接并融合
#         feature_fuse1 = self.fus1(torch.cat([wrap_gr.to(default_dtype), gt.to(default_dtype)], dim=1))
#         return feature_fuse1

# class MAM1(nn.Module):
#     def __init__(self, in_channel):
#         super(MAM1, self).__init__()
#         self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
#     def forward(self, x):
#         gr = x[0]
#         gt = x[1]
#         feature_fuse1 = self.fus1(torch.cat([gr, gt], dim=1))
#         return feature_fuse1
    
class STN(nn.Module):
    def __init__(self, in_channel):
        super(STN, self).__init__()
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
        return wrap_gr.to(default_dtype)

        # # 将对齐后的特征与第二个模态的特征拼接并融合
        # fuse = self.fus1(torch.cat([wrap_gr.to(default_dtype), gt.to(default_dtype)], dim=1))

        # return wrap_gr.to(default_dtype)

# # Feature Rectify Module
# class ChannelWeights(nn.Module):
#     def __init__(self, dim, reduction=1):
#         super(ChannelWeights, self).__init__()
#         self.dim = dim
#         # self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.dim * 2, self.dim * 2 // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.dim * 2 // reduction, self.dim * 2),
#             nn.Sigmoid())

#     def forward(self, x1, x2):
#         B, _, H, W = x1.shape
#         # x = torch.cat((x1, x2), dim=1)
#         # avg1 = self.avg_pool(x1).view(B, self.dim)
#         avg1 = torch.mean(x1, dim=[2, 3], keepdim=True).view(B, self.dim)
#         avg2 = torch.mean(x2, dim=[2, 3], keepdim=True).view(B, self.dim)
#         # avg2 = self.avg_pool(x2).view(B, self.dim)
#         max1 = self.max_pool(x1).view(B, self.dim)
#         max2 = self.max_pool(x2).view(B, self.dim)
#         avg = avg1+avg2
#         max = max1+max2
#         y = torch.cat((max, avg), dim=1)  # B 4C
#         y = self.mlp(y).view(B, self.dim * 2, 1)
#         channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
#         return channel_weights


# class SpatialWeights(nn.Module):
#     def __init__(self, dim, reduction=1):
#         super(SpatialWeights, self).__init__()
#         self.dim = dim
#         self.mlp = nn.Sequential(
#             nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
#             nn.Sigmoid())

#     def forward(self, x1, x2):
#         B, _, H, W = x1.shape
#         x = torch.cat((x1, x2), dim=1)  # B 2C H W
#         spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
#         return spatial_weights
    
# class FRM(nn.Module):
#     def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
#         super(FRM, self).__init__()
#         self.lambda_c = lambda_c
#         self.lambda_s = lambda_s
#         self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
#         self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x1, x2):
#         channel_weights = self.channel_weights(x1, x2)
#         # out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
#         x1 = x1 + self.lambda_c * channel_weights[0] * x1
#         # out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
#         x2 = x2 + self.lambda_c * channel_weights[1] * x2
#         spatial_weights = self.spatial_weights(x1, x2)
#         out_x1 = x1 + self.lambda_s * spatial_weights[0] * x1
#         out_x2 = x2 + self.lambda_s * spatial_weights[1] * x2
#         out = out_x1 + out_x2
#         return out

# class DynamicConvFusion(nn.Module):
#     def __init__(self, dim, reduction=4, lambda_c=0.5, lambda_s=0.5):
#         super(DynamicConvFusion, self).__init__()
#         self.lambda_c = lambda_c
#         self.lambda_s = lambda_s
        
#         # 特征提取模块
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(dim*2, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True)
#         )
        
#         # 动态卷积参数生成器
#         self.conv_weight_generator = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, dim//reduction, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim//reduction, 9*dim, kernel_size=1, bias=False)  # 3x3卷积核参数
#         )
        
#         # 注意力权重生成器
#         self.attention_generator = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, dim//reduction, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim//reduction, 2, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )
        
#         # 初始化权重
#         self.apply(self._init_weights)
        
#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, 0, 0.01)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
                
#     def forward(self, x1, x2):
#         # 拼接特征
#         concat_features = torch.cat([x1, x2], dim=1)
#         shared_features = self.feature_extractor(concat_features)
        
#         # 生成动态卷积权重
#         conv_weights = self.conv_weight_generator(shared_features)
#         B, C, H, W = x1.size()
#         conv_weights = conv_weights.view(B, C, 9, 1, 1)  # 重塑为[B, C, 9, 1, 1]
        
#         # 生成注意力权重
#         attn_weights = self.attention_generator(shared_features)
#         attn_x1, attn_x2 = attn_weights[:, 0:1, :, :], attn_weights[:, 1:2, :, :]
        
#         # 应用动态卷积
#         x1_reshaped = x1.reshape(1, B*C, H, W)
#         x2_reshaped = x2.reshape(1, B*C, H, W)
        
#         # 使用分组卷积实现动态卷积
#         dynamic_x1 = F.conv2d(
#             x1_reshaped, 
#             weight=conv_weights.reshape(B*C, 1, 3, 3), 
#             bias=None, 
#             stride=1, 
#             padding=1, 
#             groups=B*C
#         ).view(B, C, H, W)
        
#         dynamic_x2 = F.conv2d(
#             x2_reshaped, 
#             weight=conv_weights.reshape(B*C, 1, 3, 3), 
#             bias=None, 
#             stride=1, 
#             padding=1, 
#             groups=B*C
#         ).view(B, C, H, W)
        
#         # 加权融合
#         out_x1 = x1 + self.lambda_c * attn_x1 * dynamic_x1
#         out_x2 = x2 + self.lambda_c * attn_x2 * dynamic_x2
        
#         # 最终融合
#         out = self.lambda_s * out_x1 + (1-self.lambda_s) * out_x2
#         return out

# from torchvision.ops import deform_conv2d
# class DeformableFRM(nn.Module):
#     def __init__(self, dim, reduction=1, lambda_c=0.5):
#         super(DeformableFRM, self).__init__()
#         self.lambda_c = lambda_c
        
#         # 通道注意力模块保持不变
#         self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        
#         # 可变形卷积参数
#         self.deform_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.offset_conv = nn.Conv2d(dim * 2, 2 * 3 * 3, kernel_size=3, padding=1)
#         self.mask_conv = nn.Conv2d(dim * 2, 1 * 3 * 3, kernel_size=3, padding=1)
        
#         # 初始化权重
#         self._init_weights()

#     def _init_weights(self):
#         # 偏移卷积初始化为0，从零开始学习
#         nn.init.constant_(self.offset_conv.weight, 0)
#         nn.init.constant_(self.offset_conv.bias, 0)
        
#         # 掩码卷积初始化为0.5
#         nn.init.constant_(self.mask_conv.weight, 0)
#         nn.init.constant_(self.mask_conv.bias, 0.5)
        
#         # 可变形卷积正常初始化
#         nn.init.kaiming_normal_(self.deform_conv.weight, mode='fan_out', nonlinearity='relu')
#         if self.deform_conv.bias is not None:
#             nn.init.constant_(self.deform_conv.bias, 0)

#     def forward(self, x1, x2):
#         # 通道注意力
#         channel_weights = self.channel_weights(x1, x2)
#         x1_c = x1 + self.lambda_c * channel_weights[0] * x1
#         x2_c = x2 + self.lambda_c * channel_weights[1] * x2
        
#         # 拼接特征用于生成偏移和掩码
#         concat_features = torch.cat([x1_c, x2_c], dim=1)
        
#         # 生成偏移量 (batch_size, 2*3*3, H, W)
#         offset = self.offset_conv(concat_features)
        
#         # 生成掩码 (batch_size, 1*3*3, H, W)
#         mask = torch.sigmoid(self.mask_conv(concat_features))
        
#         # 应用可变形卷积融合特征
#         fused_features = deform_conv2d(
#             x1_c, 
#             offset, 
#             self.deform_conv.weight, 
#             self.deform_conv.bias,
#             padding=1,
#             mask=mask
#         )
        
#         # 最终融合
#         out = fused_features + x2_c
#         return out

### 带旋转的对齐模块  (MCD alignment) ###
class MCD_alignment(nn.Module):
    def __init__(self, in_channel):
        super(MCD_alignment, self).__init__()
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
        # 直接预测完整的尺度变换矩阵
        self.scale_matrix = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 4, 1, 1, 0)
        )
        # 预测旋转角度
        self.rotation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 1, 1, 1, 0)
        )
        # Start with identity transformation
        self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.xy[-1].bias.data.zero_()
        self.scale_matrix[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale_matrix[-1].bias.data.zero_()
        self.rotation[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.rotation[-1].bias.data.zero_()
        self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)

    def forward(self, x):
        gr = x[0]
        gt = x[1]
        in_ = torch.cat([gr, gt], dim=1)
        n1 = self.channel264(in_)
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
        if in_.is_cuda:
            identity_theta = identity_theta.cuda().detach()
        shift_xy = self.xy(n1)
        scale_params = self.scale_matrix(n1).squeeze()
        rotation_angle = self.rotation(n1).squeeze()
        # 检查 rotation_angle 是否为零维张量
        if rotation_angle.dim() == 0:
            rotation_angle = rotation_angle.unsqueeze(0)

        bsize = shift_xy.shape[0]
        identity_theta = identity_theta.view(-1, 2, 3).repeat(bsize, 1, 1)
        identity_theta[:, :, 2] += shift_xy.squeeze()

        # 构建尺度变换矩阵
        scale_matrix = scale_params.view(bsize, 2, 2) + torch.eye(2, device=scale_params.device).unsqueeze(0)

        # 构建旋转矩阵
        cos_theta = torch.cos(rotation_angle).unsqueeze(1).unsqueeze(2)
        sin_theta = torch.sin(rotation_angle).unsqueeze(1).unsqueeze(2)
        rotation_matrix = torch.cat([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).view(bsize, 2, 2)

        # 合并尺度和旋转矩阵
        transform_matrix = torch.bmm(scale_matrix.half(), rotation_matrix.half())

        # 更新仿射变换矩阵
        identity_theta[:, :, :2] = transform_matrix

        identity_theta = identity_theta.half()
        wrap_grid = F.affine_grid(identity_theta.view(-1, 2, 3), in_.size(), align_corners=True).permute(0, 3, 1, 2)

        wrap_gr = F.grid_sample(gr.float(), wrap_grid.float().permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

        # 获取模型当前的默认数据类型
        default_dtype = next(self.fus1.parameters()).dtype

        # 将对齐后的特征与第二个模态的特征拼接并融合
        fuse = self.fus1(torch.cat([wrap_gr.to(default_dtype), gt.to(default_dtype)], dim=1))

        return fuse

###解耦密集融合（Decoupled Dense Fusion, DDF）###
class ChannelDecoupling(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelDecoupling, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim * 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        avg1 = torch.mean(x1, dim=[2, 3], keepdim=True).view(B, self.dim)
        avg2 = torch.mean(x2, dim=[2, 3], keepdim=True).view(B, self.dim)
        max1 = F.adaptive_max_pool2d(x1, 1).view(B, self.dim)
        max2 = F.adaptive_max_pool2d(x2, 1).view(B, self.dim)
        avg = avg1 + avg2
        max = max1 + max2
        y = torch.cat((max, avg), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class SpatialDecoupling(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialDecoupling, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights


class DDF(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super(DDF, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_decoupling = ChannelDecoupling(dim=dim, reduction=reduction)
        self.spatial_decoupling = SpatialDecoupling(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, torch.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        # 通道解耦
        channel_weights = self.channel_decoupling(x1, x2)
        x1_c = x1 + self.lambda_c * channel_weights[0] * x1
        x2_c = x2 + self.lambda_c * channel_weights[1] * x2

        # 空间解耦
        spatial_weights = self.spatial_decoupling(x1_c, x2_c)
        x1_s = x1_c + self.lambda_s * spatial_weights[0] * x1_c
        x2_s = x2_c + self.lambda_s * spatial_weights[1] * x2_c

        # 最终融合
        out = x1_s + x2_s
        return out

    
###复杂解耦密集融合####
class FeaturePool(nn.Module):
    """特征池化（借鉴原Decouple的mlp_pool思路，聚合空间信息）"""
    def __init__(self, dim, reduction=2):
        super(FeaturePool, self).__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 空间维度压缩为1x1
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1)
        )

    def forward(self, x):
        return self.pool(x).squeeze(-1).squeeze(-1)  # 输出形状：[B, C]


class ChannelAttention(nn.Module):
    """通道注意力（用于细化交互部分的通道权重）"""
    def __init__(self, dim):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(self.avg_pool(x))


class SpatialAttention(nn.Module):
    """空间注意力（用于细化交互部分的空间权重）"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                              padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
        x_cat = torch.cat([avg_pool, max_pool], dim=1)  # [B,2,H,W]
        return self.sigmoid(self.conv(x_cat))  # [B,1,H,W]


class ChannelDecoupling1(nn.Module):
    """通道维度解耦：生成通道级交互门和独立门"""
    def __init__(self, dim, reduction=2):
        super(ChannelDecoupling1, self).__init__()
        self.dim = dim
        self.pool = FeaturePool(dim, reduction)  # 聚合空间信息，提取通道特征
        self.T_c = nn.Parameter(torch.ones([]) * dim)  # 可学习缩放因子

    def forward(self, x1, x2):
        B, C, H, W = x1.shape  # C = self.dim
        
        # 1. 通道特征聚合与归一化
        x1_pool = self.pool(x1)  # [B, C]
        x2_pool = self.pool(x2)  # [B, C]
        x1_norm = x1_pool / x1_pool.norm(dim=1, keepdim=True)  # 归一化，[B, C]
        x2_norm = x2_pool / x2_pool.norm(dim=1, keepdim=True)  # 归一化，[B, C]
        
        # 2. 计算通道相似度矩阵 [B, C, C]（关键修正）
        # 调整维度：x1_norm -> [B, C, 1]；x2_norm -> [B, 1, C]
        x1_mat = x1_norm.unsqueeze(2)  # [B, C, 1]
        x2_mat = x2_norm.unsqueeze(1)  # [B, 1, C]
        # 矩阵乘法得到 [B, C, C] 的相似度矩阵
        sim_matrix = torch.matmul(x1_mat, x2_mat)  # [B, C, C]
        # 乘以缩放因子，然后提取对角线元素（每个通道的自关联度），得到 [B, C]
        logits_per = self.T_c * torch.diagonal(sim_matrix, dim1=1, dim2=2)  # [B, C]
        
        # 3. 生成门控（此时logits_per形状为[B, C]，可正确reshape）
        cross_gate = torch.sigmoid(logits_per).view(B, C, 1, 1)  # [B, C, 1, 1]
        add_gate = torch.ones_like(cross_gate) - cross_gate  # [B, C, 1, 1]
        
        return cross_gate, add_gate


class SpatialDecoupling1(nn.Module):
    """空间维度解耦：生成空间级交互门和独立门，并细化交互部分"""
    def __init__(self, dim, kernel_size=7):
        super(SpatialDecoupling1, self).__init__()
        self.dim = dim
        # 深度卷积（增强空间关联性）+ 空间注意力（细化空间权重）
        self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=kernel_size, 
                               padding=kernel_size//2, groups=dim)
        self.sa = SpatialAttention(kernel_size)  # 空间注意力

    def forward(self, x1, x2, cross_gate):
        B, C, H, W = x1.shape
        
        # 1. 用通道交互门筛选出需要空间交互的特征
        x1_inter = x1 * cross_gate  # [B,C,H,W]（通道交互部分）
        x2_inter = x2 * cross_gate  # [B,C,H,W]（通道交互部分）
        
        # 2. 融合后通过卷积+注意力生成空间交互门
        x_cat = torch.cat([x1_inter, x2_inter], dim=1)  # [B,2C,H,W]
        x_fuse = self.dwconv(x_cat)  # 增强空间关联 [B,2C,H,W]
        spatial_gate = self.sa(x_fuse[:, :C, :, :] + x_fuse[:, C:, :, :])  # 空间交互门 [B,1,H,W]
        spatial_indep = torch.ones_like(spatial_gate) - spatial_gate  # 空间独立门 [B,1,H,W]
        
        return spatial_gate, spatial_indep


class gateDDF(nn.Module):
    """改写后的解耦密集融合：输出交互部分（I）和独立部分（C）"""
    def __init__(self, dim, reduction=2, kernel_size=7):
        super(gateDDF, self).__init__()
        self.dim = dim
        # 通道解耦（生成通道级交互/独立门）
        self.channel_decouple = ChannelDecoupling1(dim, reduction)
        # 空间解耦（生成空间级交互/独立门，并细化）
        self.spatial_decouple = SpatialDecoupling1(dim, kernel_size)
        # 通道注意力（进一步细化交互部分的通道权重）
        self.cse = ChannelAttention(dim*2)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        
        ###########################################
        # 步骤1：通道解耦 -> 区分通道交互/独立部分
        ###########################################
        # 生成通道门控：cross_gate（交互）、add_gate（独立）
        cross_gate, add_gate = self.channel_decouple(x1, x2)
        
        # 初步拆分：通道交互部分（用cross_gate筛选）和通道独立部分（用add_gate筛选）
        x1_c_inter = x1 * cross_gate  # x1的通道交互特征
        x2_c_inter = x2 * cross_gate  # x2的通道交互特征
        x1_c_indep = x1 * add_gate    # x1的通道独立特征
        x2_c_indep = x2 * add_gate    # x2的通道独立特征
        
        ###########################################
        # 步骤2：空间解耦 -> 细化交互部分的空间权重
        ###########################################
        # 生成空间门控：spatial_gate（交互）、spatial_indep（独立）
        spatial_gate, spatial_indep = self.spatial_decouple(x1, x2, cross_gate)
        
        # 用通道注意力进一步优化交互特征的通道权重
        x_cat = torch.cat([x1_c_inter, x2_c_inter], dim=1)  # [B,2C,H,W]
        fuse_gate = torch.sigmoid(self.cse(x_cat))  # [B,2C,1,1]
        x1_fuse_gate = fuse_gate[:, :C, :, :]  # x1的交互通道权重
        x2_fuse_gate = fuse_gate[:, C:, :, :]  # x2的交互通道权重
        
        # 最终交互部分：通道交互特征 * 空间交互门 * 优化后的通道权重
        x1_I = x1_c_inter * spatial_gate * x1_fuse_gate  # x1的最终交互特征
        x2_I = x2_c_inter * spatial_gate * x2_fuse_gate  # x2的最终交互特征
        
        # 最终独立部分：通道独立特征 * 空间独立门（不参与跨模态交互）
        x1_C = x1_c_indep * spatial_indep  # x1的最终独立特征
        x2_C = x2_c_indep * spatial_indep  # x2的最终独立特征
        
        x_cat = torch.cat([x1_I, x1_C, x2_I, x2_C], dim=1)
        return x_cat  # 返回交互部分和独立部分的融合特征 [B, 4C, H, W]
    
class VTFF_gate(nn.Module):
    # 只有gateDDF
    def __init__(self, in_channel):
        super(VTFF_gate, self).__init__()
        self.fuse = gateDDF(in_channel)
    def forward(self,x):
        rgb = x[0]
        t = x[1]
        final = self.fuse(rgb, t)
        return final
    
# class PerspectiveSTN(nn.Module):
#     def __init__(self, in_channel):
#         super(PerspectiveSTN, self).__init__()
#         self.channel264 = nn.Sequential(
#             Conv(in_channel*2, 128, 3, 2, 1),
#             Conv(128, 128, 1, 1, 1),
#             convblock(128, 64, 3, 2, 1),
#             convblock(64, 64, 1, 1, 0),
#             convblock(64, 32, 3, 2, 1),
#             convblock(32, 32, 1, 1, 0),
#         )
#         # self.channel264 = nn.Sequential(
#         #     Conv(in_channel*2, 256, 3, 2, 1),  # 通道数从128→256
#         #     nn.ReLU(inplace=True),  # 增加激活函数
#         #     Conv(256, 128, 1, 1, 0),  # 1×1卷积padding设为0（避免无效边缘）
#         #     nn.ReLU(inplace=True),
#         #     convblock(128, 128, 3, 2, 1),  # 增加中间层通道数
#         #     convblock(128, 64, 1, 1, 0),
#         #     convblock(64, 64, 3, 2, 1),
#         #     convblock(64, 32, 1, 1, 0),
#         # )
#         # 修改为预测 8 个参数（单应性矩阵的前 8 个元素）
#         self.homo = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(32, 8, 1, 1, 0)
#         )
#         # 初始化为单位变换
#         self.homo[-1].weight.data.normal_(mean=0.0, std=1e-5)
#         self.homo[-1].bias.data.zero_()
#         self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)

#     def forward(self, x):
#         gr = x[0]  # 假设形状为 (bsize, channels, h, w)
#         gt = x[1]
#         in_ = torch.cat([gr, gt], dim=1)
#         n1 = self.channel264(in_)
#         print("gr mean:", gr.mean().item(), "gt mean:", gt.mean().item())
#         print("n1 mean:", n1.mean().item())  # 若接近0，说明特征提取失败

#         # 预测单应性矩阵前8个参数
#         homo_params = self.homo(n1).view(-1, 8)
#         bsize, _, h, w = gr.shape  # 正确获取高度h和宽度w（注意这里是gr.shape[2]和gr.shape[3]）
#         print("homo_params mean:", homo_params.mean().item())

#         # 构造3×3单应性矩阵（h8=1）
#         identity_homo = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float)
#         identity_homo = identity_homo.view(1, -1).repeat(bsize, 1).to(in_.device)
#         homo_matrix = identity_homo + homo_params  # 形状：(bsize, 8)
#         homo_matrix = torch.cat([homo_matrix, torch.ones(bsize, 1).to(in_.device)], dim=1)
#         homo_matrix = homo_matrix.view(bsize, 3, 3)  # 形状：(bsize, 3, 3)

#         # 生成透视变换网格（修正维度操作）
#         # 1. 生成标准化x坐标网格 (bsize, h, w)
#         x_coords = torch.linspace(-1, 1, w, device=in_.device)  # 1D张量: (w,)
#         x_coords = x_coords.repeat(bsize, h, 1)  # 扩展为: (bsize, h, w)

#         # 2. 生成标准化y坐标网格 (bsize, h, w)
#         # 关键修正：对1D张量使用unsqueeze(1)而非unsqueeze(2)
#         y_coords = torch.linspace(-1, 1, h, device=in_.device)  # 1D张量: (h,)
#         y_coords = y_coords.unsqueeze(1)  # 变为2D张量: (h, 1)
#         y_coords = y_coords.repeat(bsize, 1, w)  # 扩展为: (bsize, h, w)

#         # 3. 构造齐次坐标网格
#         ones = torch.ones_like(x_coords)  # (bsize, h, w)
#         grid = torch.stack([x_coords, y_coords, ones], dim=3)  # (bsize, h, w, 3)

#         # 4. 应用单应性矩阵进行透视变换
#         homo_grid = torch.matmul(homo_matrix.unsqueeze(1).unsqueeze(1), grid.unsqueeze(4))  # (bsize, h, w, 3, 1)
#         homo_grid = homo_grid.squeeze(4)  # (bsize, h, w, 3)
#         epsilon = 1e-8
#         x_transformed = homo_grid[..., 0] / (homo_grid[..., 2] + epsilon)  # 透视除法
#         y_transformed = homo_grid[..., 1] / (homo_grid[..., 2] + epsilon)

#         x_transformed = torch.clamp(x_transformed, -1.0, 1.0)
#         y_transformed = torch.clamp(y_transformed, -1.0, 1.0)
#         wrap_grid = torch.stack([x_transformed, y_transformed], dim=3)  # (bsize, h, w, 2)

#         # 5. 采样得到变换后的特征
#         wrap_gr = F.grid_sample(
#             gr.float(), 
#             wrap_grid.float(), 
#             mode='bilinear', 
#             padding_mode='zeros', 
#             align_corners=True
#         )
#         print("wrap_grid range:", wrap_grid.min().item(), wrap_grid.max().item())  # 应在[-1,1]附近
#         print("wrap_gr mean:", wrap_gr.mean().item())  # 若为0，说明变换破坏了特征
#         # 恢复默认数据类型
#         default_dtype = next(self.parameters()).dtype
#         return wrap_gr.to(default_dtype)
#     # def forward(self, x):
#     #     gr = x[0]
#     #     gt = x[1]
#     #     in_ = torch.cat([gr, gt], dim=1)
#     #     n1 = self.channel264(in_)

#     #     # 预测单应性矩阵的前 8 个元素
#     #     homo_params = self.homo(n1).view(-1, 8)
#     #     bsize = homo_params.shape[0]

#     #     # 构造单应性矩阵
#     #     identity_homo = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float).view(1, -1).repeat(bsize, 1)
#     #     if in_.is_cuda:
#     #         identity_homo = identity_homo.cuda()
#     #     homo_matrix = identity_homo + homo_params
#     #     # homo_matrix = torch.cat([homo_matrix, torch.tensor([[0, 0, 1]]).repeat(bsize, 1).to(homo_matrix.device)], dim=1).view(bsize, 3, 3)
#     #     homo_matrix = torch.cat([homo_matrix, torch.ones(bsize, 1).to(homo_matrix.device)], dim=1).view(bsize, 3, 3)

#     #     # 生成网格并进行采样
#     #     wrap_grid = F.affine_grid(homo_matrix[:, :2, :], in_.size(), align_corners=True).permute(0, 3, 1, 2)
#     #     wrap_gr = F.grid_sample(gr.float(), wrap_grid.float().permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

#     #     # 获取模型当前的默认数据类型
#     #     default_dtype = next(self.fus1.parameters()).dtype

#     #     return wrap_gr.to(default_dtype)

class PerspectiveSTN(nn.Module):
    def __init__(self, in_channel):
        super(PerspectiveSTN, self).__init__()
        # 共享特征提取 backbone
        self.shared_backbone = nn.Sequential(
            Conv(in_channel*2, 128, 3, 2, 1),
            Conv(128, 128, 1, 1, 1),  # 修正1x1卷积的padding为0（原1可能不合理）
            convblock(128, 64, 3, 2, 1),
            convblock(64, 64, 1, 1, 0),
            convblock(64, 32, 3, 2, 1),
            convblock(32, 32, 1, 1, 0),
        )
        
        # 子网络1：预测单应性矩阵前4个参数 [h0, h1, h2, h3]
        self.homo_subnet1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 4, 1, 1, 0)  # 输出4个参数
        )
        
        # 子网络2：预测单应性矩阵后4个参数 [h4, h5, h6, h7]
        self.homo_subnet2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 4, 1, 1, 0)  # 输出4个参数
        )
        
        # 初始化：所有子网络输出接近0，确保初始为单位变换
        for subnet in [self.homo_subnet1, self.homo_subnet2]:
            subnet[-1].weight.data.normal_(mean=0.0, std=1e-5)  # 极小初始化
            subnet[-1].bias.data.zero_()

    def forward(self, x):
        gr, gt = x[0], x[1]  # 输入应为两个特征图的列表/元组
        in_ = torch.cat([gr, gt], dim=1)  # 特征拼接 (bs, 2*C, H, W)
        bsize, _, h, w = gr.shape  # 获取输入特征尺寸
        
        # 1. 共享特征提取
        shared_feat = self.shared_backbone(in_)  # (bs, 32, H', W')
        
        # 2. 多子网络预测参数（齐次坐标参数分散学习）
        params1 = self.homo_subnet1(shared_feat).view(bsize, 4)  # [h0, h1, h2, h3]
        params2 = self.homo_subnet2(shared_feat).view(bsize, 4)  # [h4, h5, h6, h7]
        homo_params = torch.cat([params1, params2], dim=1)  # 拼接为8个参数 (bs, 8)
        
        # 3. 构造3x3单应性矩阵（h8固定为1）
        identity = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float, device=in_.device)
        identity = identity.repeat(bsize, 1)  # 单位矩阵初始值 (bs, 8)
        homo_matrix = identity + homo_params  # 叠加预测的偏移量
        homo_matrix = torch.cat([homo_matrix, torch.ones(bsize, 1, device=in_.device)], dim=1)
        homo_matrix = homo_matrix.view(bsize, 3, 3)  # 转换为3x3矩阵 (bs, 3, 3)
        
        # 4. 生成透视变换网格（带数值稳定措施）
        # 4.1 生成标准化坐标网格 [-1, 1]
        x_coords = torch.linspace(-1, 1, w, device=in_.device)  # (w,)
        x_coords = x_coords.repeat(bsize, h, 1)  # (bs, h, w)
        
        y_coords = torch.linspace(-1, 1, h, device=in_.device)  # (h,)
        y_coords = y_coords.unsqueeze(1).repeat(bsize, 1, w)  # (bs, h, w)
        
        ones = torch.ones_like(x_coords)  # (bs, h, w)
        grid = torch.stack([x_coords, y_coords, ones], dim=3)  # (bs, h, w, 3)
        
        # 4.2 应用单应性矩阵（透视变换）
        homo_grid = torch.matmul(homo_matrix.unsqueeze(1).unsqueeze(1), grid.unsqueeze(4))  # (bs, h, w, 3, 1)
        homo_grid = homo_grid.squeeze(4)  # (bs, h, w, 3)
        
        # 4.3 透视除法（添加epsilon避免除零）
        epsilon = 1e-8
        x_transformed = homo_grid[..., 0] / (homo_grid[..., 2] + epsilon)
        y_transformed = homo_grid[..., 1] / (homo_grid[..., 2] + epsilon)
        
        # 4.4 约束坐标范围（避免超出有效区域）
        x_transformed = torch.clamp(x_transformed, -1.0, 1.0)
        y_transformed = torch.clamp(y_transformed, -1.0, 1.0)
        wrap_grid = torch.stack([x_transformed, y_transformed], dim=3)  # (bs, h, w, 2)
        
        # 5. 网格采样得到变换后的特征
        wrap_gr = F.grid_sample(
            gr.float(),
            wrap_grid.float(),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # 恢复原始数据类型
        return wrap_gr.to(gr.dtype)


class VTFFA(nn.Module):
    def __init__(self, in_channel):
        super(VTFFA, self).__init__()
        self.alignment = MCD_alignment(in_channel)
        self.fuse = DDF(in_channel)
    def forward(self,x):
        # rgb = x[0]
        t = x[1]
        # x = [t, rgb]
        gr = self.alignment(x)
        final = self.fuse(gr, t)
        return final

class VTFFA1(nn.Module):
    # 直接将对齐后的特征与时间模态特征相加
    # VTFFA without DDF
    def __init__(self, in_channel):
        super(VTFFA1, self).__init__()
        self.alignment = MCD_alignment(in_channel)
    def forward(self,x):
        # rgb = x[0]
        t = x[1]
        # x = [t, rgb]
        gr = self.alignment(x)
        final = gr + t
        return final
    
class VTFFA2(nn.Module):
    # 不对齐直接融合
    # VTFFA without MCD_alignment
    def __init__(self, in_channel):
        super(VTFFA2, self).__init__()
        self.fuse = DDF(in_channel)
    def forward(self,x):
        rgb = x[0]
        t = x[1]
        final = self.fuse(rgb, t)
        return final

class PerspectiveSTN_DDF(nn.Module):
    def __init__(self, in_channel):
        super(PerspectiveSTN_DDF, self).__init__()
        self.alignment = PerspectiveSTN(in_channel)
        self.fuse = DDF(in_channel)
    def forward(self,x):
        # rgb = x[0]
        t = x[1]
        # x = [t, rgb]
        gr = self.alignment(x)
        final = self.fuse(gr, t)
        return final
    
class PerspectiveSTN_gateDDF(nn.Module):
    def __init__(self, in_channel):
        super(PerspectiveSTN_DDF, self).__init__()
        self.alignment = PerspectiveSTN(in_channel)
        self.fuse = gateDDF(in_channel)
    def forward(self,x):
        # rgb = x[0]
        t = x[1]
        # x = [t, rgb]
        gr = self.alignment(x)
        final = self.fuse(gr, t)
        return final
    
class STN_DDF(nn.Module):
    def __init__(self, in_channel):
        super(STN_DDF, self).__init__()
        self.alignment = STN(in_channel)
        self.fuse = DDF(in_channel)
    def forward(self,x):
        # rgb = x[0]
        t = x[1]
        # x = [t, rgb]
        gr = self.alignment(x)
        final = self.fuse(gr, t)
        return final
    
class STN_gateDDF(nn.Module):
    def __init__(self, in_channel):
        super(STN_gateDDF, self).__init__()
        self.alignment = STN(in_channel)
        self.fuse = gateDDF(in_channel)
    def forward(self,x):
        # rgb = x[0]
        t = x[1]
        # x = [t, rgb]
        gr = self.alignment(x)
        final = self.fuse(gr, t)
        return final

class CFFA_STN(nn.Module):
    # 直接将对齐后的特征与时间模态特征相加
    def __init__(self, in_channel):
        super(CFFA_STN, self).__init__()
        self.alignment = STN(in_channel)
    def forward(self,x):
        # rgb = x[0]
        t = x[1]
        # x = [t, rgb]
        gr = self.alignment(x)
        final = gr + t
        return final
    
class CFFA_PerspectiveSTN(nn.Module):
    # 直接将对齐后的特征与时间模态特征相加
    def __init__(self, in_channel):
        super(CFFA_PerspectiveSTN, self).__init__()
        self.alignment = PerspectiveSTN(in_channel)
    def forward(self,x):
        # rgb = x[0]
        t = x[1]
        # x = [t, rgb]
        gr = self.alignment(x)
        final = gr + t
        return final