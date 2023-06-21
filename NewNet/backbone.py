import torch.nn as nn
from einops import rearrange

from NewNet.dwc import DWCBlock
from models.attentions.multi_path_attn import MultiPathAttn
from models.others.vit import ViT
from thop import profile

"""
1: 常规encoder flops: 14840.29 M, params: 54.16 M
2: se en  flops: 17415.75 M, params: 55.25 M
3: se en+de flops: 13663.04 M, params: 53.41 M
"""


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


class SENet(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(SENet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        mid_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        # SE_Block放在BN之后，shortcut之前
        self.SE = SE_Block(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        SE_out = self.SE(out)
        out = out * SE_out
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class upSkip(nn.Module):
    def __init__(self, ch_g, ch_x):
        super(upSkip, self).__init__()
        self.A_g = MultiPathAttn(channels=ch_g)
        self.A_x = MultiPathAttn(channels=ch_x)

    def forward(self, g, x):
        g = self.A_g(g)
        x = self.A_x(x)
        return g + x

class EncoderBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(EncoderBottleneck, self).__init__()
        self.model = SENet(in_ch, out_ch, stride)

    def forward(self, x):
        return self.model(x)


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gate = upSkip(in_channels, in_channels)
        self.layer = SENet(in_channels, out_channels)

    def forward(self, x, g_skip=None):
        x = _upsample_like(x, g_skip)
        if g_skip is not None:
            x = self.gate(g_skip, x)
        x = self.layer(x)
        return x


def _upsample_like(src, tar):
    src = nn.functional.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src


class NewNet(nn.Module):
    def __init__(self, img_dim, in_channels, class_num, out_channels=128, head_num=4, mlp_dim=512, block_num=8,
                 patch_dim=16):
        super().__init__()
        self.encoder1 = EncoderBottleneck(in_channels, out_channels, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder4 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.decoder1 = DecoderBottleneck(out_channels, out_channels // 2)
        self.decoder2 = DecoderBottleneck(out_channels * 2, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 4, out_channels * 2)
        self.decoder4 = DecoderBottleneck(out_channels * 8, out_channels * 4)

        self.side1 = nn.Conv2d(out_channels // 2, class_num, 3, padding=1)
        self.side2 = nn.Conv2d(out_channels, class_num, 3, padding=1)
        self.side3 = nn.Conv2d(out_channels * 2, class_num, 3, padding=1)

        self.outconv = nn.Conv2d(3 * class_num, class_num, 1)

    def forward(self, x):
        g1 = self.encoder1(x)
        g2 = self.encoder2(g1)
        g3 = self.encoder3(g2)
        g4 = self.encoder4(g3)

        g_vit = self.vit(g4)
        g_vit = rearrange(g_vit, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x4d = self.decoder4(g_vit, g4)
        x3d = self.decoder3(x4d, g3)
        x2d = self.decoder2(x3d, g2)
        x1d = self.decoder1(x2d, g1)

        d1 = _upsample_like(self.side1(x1d), x)
        d2 = _upsample_like(self.side2(x2d), x)
        d3 = _upsample_like(self.side3(x3d), x)

        d0 = self.outconv(torch.cat((d1, d2, d3), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3)


if __name__ == '__main__':
    import torch

    model = NewNet(img_dim=416,
                   in_channels=3,
                   out_channels=64,
                   head_num=4,
                   mlp_dim=512,
                   block_num=8,
                   patch_dim=16,
                   class_num=2)

    dummy_input = torch.randn(1, 3, 416, 416)

    res = model(dummy_input)
    print(res[0].size())

    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
