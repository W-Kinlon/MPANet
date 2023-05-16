from u2net import _upsample_like
from u2net import *


# AgU2Net
class AgU2Net(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(AgU2Net, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.attn5 = AttentionBlock(512, 512, 256)
        self.stage5d = RSU4F(1024, 256, 512)
        self.attn4 = AttentionBlock(512, 512, 256)
        self.stage4d = RSU4(1024, 128, 256)
        self.attn3 = AttentionBlock(256, 256, 128)
        self.stage3d = RSU5(512, 64, 128)
        self.attn2 = AttentionBlock(128, 128, 64)
        self.stage2d = RSU6(256, 32, 64)
        self.attn1 = AttentionBlock(64, 64, 32)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(self.attn5(hx6up, hx5))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(self.attn4(hx5dup, hx4))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(self.attn3(hx4dup, hx3))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(self.attn2(hx3dup, hx2))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(self.attn1(hx2dup, hx1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)


if __name__ == '__main__':
    m = AgU2Net(3, 2)
    x = torch.randn(1, 3, 416, 416)
    y = m(x)
    print(y[0].shape)
    print(sum(p.numel() for p in m.parameters()))
