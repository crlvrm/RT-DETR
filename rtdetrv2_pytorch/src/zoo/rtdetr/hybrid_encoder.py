"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict
import torch.nn.init as init
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation, bias_init_with_prob

from ...core import register

# from ...nn.backbone.hgnetv2 import EMA

__all__ = ['HybridEncoder']

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))
class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = ConvNormLayer_fuse(c1 // 2, self.c, 1, 1)
        # self.cv1 = ConvNormLayer(c1 // 2, self.c, 3, 2, 1)
        self.cv3 = ConvNormLayer_fuse(self.c, self.c, 3, 2, self.c)
        self.cv2 = ConvNormLayer_fuse(c1 // 2, self.c, 1, 1)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv3(self.cv1(x1))
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvNormLayer(c1, c_, 1, 1)
        self.cv2 = ConvNormLayer(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])

        # self.gate = Gate(hidden_channels)
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        # return self.conv3(self.gate(self.attention(x_1), x_2))
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPRepLayer(c3//2, c4, n, 1, bias=bias, act=act), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPRepLayer(c4, c4, n, 1, bias=bias, act=act), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU(inplace=True) if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
# class FEM(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
#         super(FEM, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce
#         self.branch0 = nn.Sequential(
#             BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )
#
#         self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#
#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)
#
#         return out
#

class DepthwiseDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseDilatedConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvGnLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        # self.conv = nn.Conv2d(
        #     ch_in,
        #     ch_out,
        #     kernel_size,
        #     stride,
        #     groups=g,
        #     padding=padding,
        #     bias=bias)
        self.conv = DepthwiseDilatedConv(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=padding)
        self.norm = nn.GroupNorm(32, ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class CCM(nn.Module):
    def __init__(self, out_channels=256):
        super(CCM, self).__init__()
        self.ccm_cfg = [256, 256, 256]
        in_channels = 256
        self.conv1 = nn.Conv2d(64, 256, kernel_size=1)
        self.norm = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, in_channels, kernel_size=1)
        layers = []
        d_rates = [2, 2, 2]
        for i, v in enumerate(self.ccm_cfg):
            conv2d = DepthwiseDilatedConv(in_channels, v, kernel_size=3, padding=d_rates[i], dilation=d_rates[i])
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        self.ccm = nn.Sequential(*layers)
        self.conv_out1 = ConvGnLayer_fuse(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv_out2 = ConvGnLayer_fuse(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv_out3 = ConvGnLayer_fuse(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.norm(self.conv1(x))
        x = self.conv2(x)

        x = self.ccm(x)
        x_out = []
        x = self.conv_out1(x)
        x_out.append(x)
        x = self.conv_out2(x)
        x_out.append(x)
        x = self.conv_out3(x)
        x_out.append(x)
        return x_out

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = ConvNormLayer_fuse(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, act='relu')
        # self.spatial = Conv_BN(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale
class FEM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(FEM, self).__init__()

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x, feats):
        out_feats = []
        for i, feat in enumerate(feats):
            c2 = self.SpatialGate(x[i])
            feat = feat * c2
            c1 = self.ChannelGate(feat)
            feat = feat * c1
            out_feats.append(feat)

        return out_feats

class FFM_Concat2(nn.Module):
    def __init__(self, dimension=1, Channel1 = 1, Channel2 = 1):
        super(FFM_Concat2, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel_all = int(Channel1 + Channel2)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型 parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()

        w = self.w[:(C1 + C2)] # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x = [x1, x2]
        return torch.cat(x, self.d)

class FFM_Concat3(nn.Module):
    def __init__(self, dimension=1, Channel1 = 1, Channel2 = 1, Channel3 = 1):
        super(FFM_Concat3, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel3 = Channel3
        self.Channel_all = int(Channel1 + Channel2 + Channel3)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()
        N3, C3, H3, W3 = x[2].size()

        w = self.w[:(C1 + C2 + C3)]  # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:(C1 + C2)] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x3 = (weight[(C1 + C2):] * x[2].view(N3, H3, W3, C3)).view(N3, C3, H3, W3)
        x = [x1, x2, x3]
        return torch.cat(x, self.d)

class Gate(nn.Module):
    ''' 将两个输入动态相加，调整每个输入的权重 '''
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2).permute(0, 3, 1, 2).contiguous()
class AdaptiveLowLightEnhance(nn.Module):
    def __init__(self, channels, init_gamma=1.2):
        super(AdaptiveLowLightEnhance, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Estimate scene brightness
        print(x.shape)
        brightness = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        print(brightness.shape)
        brightness = torch.clamp(brightness, max=1.0)
        print(brightness.shape)
        # Control gamma and attention strength based on brightness
        gamma_adjusted = torch.pow(torch.clamp(x, min=1e-6), self.gamma * (1 - brightness))
        attention = self.fc(gamma_adjusted).expand_as(gamma_adjusted)  # 确保广播一致
        output = gamma_adjusted * attention
        return output

# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

class EfficentTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, in_channel, norm=None):
        super(EfficentTransformerEncoder, self).__init__()
        self.c = int(in_channel )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.cv2 = ConvNormLayer_fuse(2 * self.c, in_channel, 1, 1, act='silu')
        self.ffn = nn.Sequential(ConvNormLayer_fuse(self.c, self.c * 2, 1, 1, act='silu'), ConvNormLayer_fuse(self.c * 2, self.c, 1, 1))

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:

        b = src
        for layer in self.layers:
            h, w = b.shape[2:]
            src_flatten = b.flatten(2).permute(0, 2, 1)
            b = b + layer(src_flatten, src_mask=src_mask, pos_embed=pos_embed).permute(0, 2, 1).reshape(-1, self.c, h, w)
            b = b + self.ffn(b)
        return self.cv2(torch.cat((src, b), 1))
        # output = src
        # for layer in self.layers:
        #     output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        #
        # if self.norm is not None:
        #     output = self.norm(output)
        # output = output + src
        # output = output + self.ffn(output)
        # output = self.cv2(torch.cat((src, output), dim=1))
        # return output

@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2',
                 dysample=False,
                 bifpn=False):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.dysample = dysample
        self.bifpn = bifpn
        # channel projection
        self.input_proj = nn.ModuleList()
        self.atts = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)
            # self.atts.append(AdaptiveLowLightEnhance(hidden_dim))
        # self.sppf = SPPF(hidden_dim, hidden_dim)
        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        # self.encoder = nn.ModuleList([
        #     TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        # ])
        self.encoder = nn.ModuleList([
            EfficentTransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers,  hidden_dim) for _ in range(len(use_encoder_idx))
        ])
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.dys = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # 替换为融合卷积和yolov9中的GELAN
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                # RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2),
                #              round(3 * depth_mult))

            )
            self.dys.append(DySample(self.hidden_dim))
        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.conca = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.downsample_convs.append(
                # ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
                SCDown(hidden_dim, hidden_dim,3,2),
            )
            if i==0:
                self.conca.append(FFM_Concat3(Channel1 = hidden_dim, Channel2 = hidden_dim, Channel3 = hidden_dim))
                self.pan_blocks.append(
                    CSPRepLayer(hidden_dim * 3, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                    # RepNCSPELAN4(hidden_dim * 3, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2),
                    #              round(3 * depth_mult))
                )
            else:
                self.conca.append(FFM_Concat2(Channel1 = hidden_dim, Channel2 = hidden_dim))
                self.pan_blocks.append(
                    CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                    # RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2),
                    #              round(3 * depth_mult))
                )
        self.CCM = CCM()
        self.fem = FEM(gate_channels=256, reduction_ratio=16)
        # self.fem = nn.ModuleList()
        # for i in range(len(in_channels)):
        #     self.fem.append(FEM(hidden_dim, hidden_dim, stride=1, scale=0.1, map_reduce=8))
        #
        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, all_feats):
        feats = all_feats[0]
        fe_feat = all_feats[1]
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # proj_feats = [self.atts[i](feat) for i, feat in enumerate(proj_feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                # proj_feats[enc_ind] = self.sppf(proj_feats[enc_ind])
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                # memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                # proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                memory: torch.Tensor = self.encoder[i](proj_feats[enc_ind], pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = self.dys[len(self.in_channels) - 1 - idx](feat_heigh)
            # upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if idx==0:
                # inp = torch.concat([downsample_feat, feat_height, proj_feats[idx+1]], dim=1)
                inp = self.conca[idx]([downsample_feat, feat_height, proj_feats[idx+1]])
            else:
                # inp = torch.concat([downsample_feat, feat_height], dim=1)
                inp = self.conca[idx]([downsample_feat, feat_height])
            out = self.pan_blocks[idx](inp)
            outs.append(out)
        fe_feats = self.CCM(fe_feat)
        fe_outs = self.fem(fe_feats, outs)
        # fe_outs = []
        # for idx in range(len(self.in_channels)):
        #     fe_outs.append(self.fem[idx](outs[idx]))
        #
        return fe_outs
        # return outs
