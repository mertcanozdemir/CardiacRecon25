import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional as F


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.encoder1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU()

        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, x):
        skip_connections = []
        x = self.firstconv(x)
        skip_connections.append(x)
        x = self.encoder1(x)
        skip_connections.append(x)
        concat_image1 = skip_connections.pop()
        concat_image2 = skip_connections.pop()
        x = self.bottleneck(x)
        x = F.interpolate(x, size=concat_image1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.concat((concat_image1, x), dim=1)
        x = self.decoder1(x)
        x = F.interpolate(x, size=concat_image2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.concat((concat_image2, x), dim=1)
        x = self.decoder2(x)
        return x
