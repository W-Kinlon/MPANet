from ag_unet import *
from models.attentions.multi_path_attn import MultiPathAttn


class MPAttnGate(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(MPAttnGate, self).__init__()
        # ag-multi2
        self.front_attn = MultiPathAttn(channels=F_g)
        self.down_attn = MultiPathAttn(channels=F_g)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.front_attn(g)
        x = self.down_attn(x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class MPAUNet(nn.Module):
    def __init__(self,
                 in_channel=3,
                 num_classes=1,
                 channel_list=[64, 128, 256, 512, 1024],
                 checkpoint=False,
                 convTranspose=True):
        super(MPAUNet, self).__init__()
        self.net = AgUNet(in_channel, num_classes, channel_list, checkpoint, MPAttnGate, convTranspose)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    x = torch.rand(1, 3, 416, 416)
    m = MPAUNet(in_channel=3, num_classes=2)
    y = m(x)
    print(y.shape)
