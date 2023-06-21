import torch
from timm.layers import get_act_layer
from torch import nn

from aggregation_zeropad import LocalConvolution


class CotLayer(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, kernel_size=3):
        # 调用 CotLayer(width, kernel_size=3)
        super(CotLayer, self).__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            # 分组卷积
            # Torch.nn.Conv2d(in_channels,out_channels,kernel_size,
            #  stride,padding,dilation,groups,bias)
            nn.Conv2d(in_ch, in_ch, self.kernel_size, stride=1, padding=self.kernel_size // 2, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

        # 共享通道，感觉和缩放一样，，
        share_planes = 8
        # 缩放因子
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * in_ch, mid_ch, 3),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),


            # w_\theta, 存在concat操作
            nn.Conv2d(mid_ch, mid_ch // factor, 1, bias=False),
            nn.BatchNorm2d(mid_ch // factor),
            nn.ReLU(inplace=True),
            # w_\delta, 没有激活函数
            nn.Conv2d(mid_ch // factor, pow(kernel_size, 2) * mid_ch // share_planes, kernel_size=1),
            # 将channel方向分group，然后每个group内做归一化
            nn.GroupNorm(num_groups=mid_ch // share_planes, num_channels=pow(kernel_size, 2) * mid_ch // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(in_ch)
        )

        self.local_conv = LocalConvolution(in_ch, mid_ch, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(mid_ch)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2

        attn_chs = max(mid_ch * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(mid_ch, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, out_ch, 1)
        )

    def forward(self, x):
        # 用一个3x3卷积得到 key, 静态上下文信息 K^1
        k = self.key_embed(x)
        # concat => 2C x H x W
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        # 两个1x1卷积操作，得到注意力图
        w = self.embed(qk)
        # 转换成一维向量，自动补齐，k x k，H，W
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)

        # 计算 value
        x = self.conv1x1(x)

        # 使用注意力图（每个key对应一个k x k注意力矩阵）计算 value，得到动态上下文信息 K^2
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        # 拼接上面得到的静态和动态上下文信息
        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        # 使用一个SE注意模块进行融合
        x_gap = x.sum(dim=2)  # 求和合并(B, C, H, W)
        x_gap = x_gap.mean((2, 3), keepdim=True)  # (B, C, 1, 1)
        x_attn = self.se(x_gap)  # (B, 2C, 1, 1)
        x_attn = x_attn.view(B, C, self.radix)  # (B, C, 2)
        x_attn = torch.softmax(x_attn, dim=2)
        # (B, C, 2, H, W) * (B, C, 2, 1, 1) , (B, C, H, W)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)

        return out.contiguous()


if __name__ == '__main__':
    x = torch.randn(1, 3, 128, 128)
    m = CotLayer(3, 32, 64)
    y = m(x)
    print(y)
