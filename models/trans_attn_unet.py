from models.ag_unet import AttnGate
from models.trans_unet import nn,Encoder


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DecoderBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attn = AttnGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x_concat = self.attn(g=x, x=x_concat)
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super(Decoder, self).__init__()
        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.attn1 = AttnGate(F_g=out_channels * 2, F_l=out_channels * 2, F_int=out_channels)

        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, _x1, _x2, _x3):
        x1 = self.decoder1(x, _x3)
        x2 = self.decoder2(x1, _x2)
        x3 = self.decoder3(x2, _x1)
        x4 = self.decoder4(x3)
        return self.conv1(x4)


class TransAttnUNet(nn.Module):
    def __init__(self, img_dim, in_channels, class_num, out_channels=128, head_num=4, mlp_dim=512, block_num=8,
                 patch_dim=16):
        super(TransAttnUNet, self).__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels, class_num)

        self.last = nn.Sigmoid()

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        x = self.last(x)

        return x


if __name__ == '__main__':
    import torch

    transunet = TransAttnUNet(img_dim=416, in_channels=3, class_num=2)

    # print(sum(p.numel() for p in transunet.parameters()))
    res = transunet(torch.randn(1, 3, 416, 416))
    print(res.shape)
