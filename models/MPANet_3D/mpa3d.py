import torch
import torch.nn as nn


class Mpa3d(nn.Module):
    def __init__(self, channels, reduction_ratio=16, num_paths=4):
        super(Mpa3d, self).__init__()

        self.num_paths = num_paths

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1),
        )
        # print("Shared MLP weight shape:", list(self.shared_mlp.parameters())[0].shape)

        self.path_mlp = nn.ModuleList()
        for _ in range(num_paths):
            self.path_mlp.append(nn.Sequential(
                nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1),
            ))
            # print("Path", _+1, "MLP weight shape:", list(self.path_mlp[-1].parameters())[0].shape)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Conv3d(channels * 10, channels * 5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels * 5, channels, kernel_size=1),
        )
        # print("FC weight shape:", list(self.fc.parameters())[0].shape)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, channels, height, width, slices = x.size()

        # Shared MLP
        shared_out = self.shared_mlp(x)

        # Path-specific MLPs
        path_out = []
        for i in range(self.num_paths):
            path_out.append(self.path_mlp[i](x))

        # Pooling and concatenation
        avg_out = self.avgpool(shared_out)
        max_out = self.maxpool(shared_out)

        # Concatenate pooled features from all paths
        for i in range(self.num_paths):
            avg_out = torch.cat([avg_out, self.avgpool(path_out[i])], dim=1)
            max_out = torch.cat([max_out, self.maxpool(path_out[i])], dim=1)

        # Channel-wise attention
        a = torch.cat([avg_out, max_out], dim=1)
        channel_attention = self.sigmoid(self.fc(a))
        # print("Channel attention weight shape:", list(self.fc.parameters())[0].shape)

        # Spatial attention
        spatial_attention = torch.sigmoid(torch.mean(shared_out, dim=1, keepdim=True))

        # Apply attention
        out = channel_attention * shared_out * spatial_attention

        return out


if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 4, 4, 20)
    multi_attn = Mpa3d(channels=64)
    output_tensor = multi_attn(input_tensor)
    print(output_tensor.size())
    print(sum(p.numel() for p in multi_attn.parameters()))
