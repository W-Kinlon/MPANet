import torch
from timm.layers import DropPath
from torch import nn


class DWCBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # # 常规放大通道
        # self.cbr = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False),
        #     nn.BatchNorm2d(out_ch)
        # )

        # 分组卷积+大卷积核
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # 在1x1之前使用唯一一次LN做归一化
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # 全连接层跟1x1conv等价，但pytorch计算上fc略快
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # 整个block只使用唯一一次激活层
        self.act = nn.GELU()
        # 反瓶颈结构，中间层升维了4倍
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # gamma的作用是用于做layer scale训练策略
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        # drop_path是用于stoch. depth训练策略
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x = self.cbr(x)
        x = self.dwconv(x)
        # 由于用FC来做1x1conv，所以需要调换通道顺序
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return self.drop_path(x)


if __name__ == '__main__':
    x = torch.randn(1, 32, 128, 128)  # 131,072
    m = DWCBlock(32)
    y = m(x)
    print(y.size())
    print(sum(p.numel() for p in m.parameters()))
