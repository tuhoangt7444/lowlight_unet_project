import torch
import torch.nn as nn


class CABlock(nn.Module):
    """
    Channel Attention đơn giản:
    - Global Avg Pool
    - MLP nhỏ
    - Sigmoid
    - Nhân lại với feature map
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """
    2 lớp Conv + ReLU + attention
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CABlock(out_ch),
        )

    def forward(self, x):
        return self.conv(x)


class UNetLite(nn.Module):
    """
    U-Net Lightweight cho tăng sáng + làm rõ ảnh.
    Encoder 2 tầng, bottleneck, decoder 2 tầng, skip connections.
    """

    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)

        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)  # 64 (up) + 64 (skip)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)   # 32 (up) + 32 (skip)

        # Output
        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))

        # Decoder
        x = self.up2(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        x = self.out_conv(x)
        x = torch.sigmoid(x)  # đầu ra [0,1]
        return x
