"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init
from collections import OrderedDict

from .common import get_activation, FrozenBatchNorm2d

from ...core import register


__all__ = ['StarNet']

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}



donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Create random tensor (like a dropout mask) with the same dimensions as `x`
        random_tensor = keep_prob + torch.rand(x.shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)  # Binary mask with values 0 or 1
        # Scale the output to maintain the expected sum of the inputs
        return x / keep_prob * binary_mask

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x
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
        self.act = get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))




@register()
class StarNet(nn.Module):
    def __init__(
            self,
            name,
            variant='d',
            num_stages=4,
            return_idx=[0, 1, 2, 3],
            act='relu',
            freeze_at=-1,
            freeze_norm=True,
            pretrained=False):
        super().__init__()

        block_nums = [1, 2, 6, 2]
        mlp_ratio = 4
        drop_path_rate = 0.0
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        # self.conv1 = nn.Sequential(OrderedDict([
        #     (name, ConvNormLayer(cin, cout, k, s, act=act)) for cin, cout, k, s, name in conv_def
        # ]))
        self.conv1 = nn.Sequential(ConvBN(3, ch_in, kernel_size=3, stride=2, padding=1), nn.ReLU6())

        ch_out_list = [64, 128, 256, 512]


        _out_channels = ch_out_list
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(block_nums))]  # stochastic depth
        cur = 0
        for i_layer in range(len(block_nums)):

            down_sampler = ConvBN(ch_in, ch_out_list[i_layer], 3, 2, 1)
            blocks = [Block(ch_out_list[i_layer], mlp_ratio, dpr[cur + i]) for i in range(block_nums[i_layer])]
            self.res_layers.append(
                nn.Sequential(down_sampler, *blocks)
            )
            cur += block_nums[i_layer]
            ch_in = _out_channels[i_layer]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            if isinstance(pretrained, bool) or 'http' in pretrained:
                state = torch.hub.load_state_dict_from_url(model_urls[name], map_location='cpu')
            # else:
            #     state = torch.load(pretrained, map_location='cpu')
                self.load_state_dict(state)
                print(f'Load StarNet{name} state_dict')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs