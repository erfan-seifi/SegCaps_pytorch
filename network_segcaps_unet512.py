import torch.nn as nn
from convcapsulelayer_unet import ConvCapsuleLayer
import torch
import torch.nn.functional as F


class CapsNetU512(nn.Module):
    def __init__(self):
        super(CapsNetU512, self).__init__()

        # https://github.com/lalonderodney/SegCaps/blob/master/imgs/segcaps.png
        self.NP0, self.LP0, self.stride0 = 1, 16, 1
        self.NP1, self.LP1, self.stride1 = 2, 16, 2
        self.NP2, self.LP2, self.stride2 = 4, 16, 1
        self.NP3, self.LP3, self.stride3 = 4, 32, 2
        self.NP4, self.LP4, self.stride4 = 8, 32, 1
        self.NP5, self.LP5, self.stride5 = 8, 64, 2
        self.NP6, self.LP6, self.stride6 = 8, 32, 1
        self.NU1, self.LU1, self.stridu1 = 8, 32, 2
        self.NP7, self.LP7, self.stride7 = 4, 32, 1
        self.NU2, self.LU2, self.stridu2 = 4, 16, 2
        self.NP8, self.LP8, self.stride8 = 4, 16, 1
        self.NU3, self.LU3, self.stridu3 = 2, 16, 2
        self.NP9, self.LP9, self.stride9 = 1, 16, 1
        self.cap_1 = ConvCapsuleLayer(kernel_size=5, NP=self.NP1, LC=self.LP0, LP=self.LP1, strides=self.stride1, padding='same', routings=1, deconv=False)
        self.cap_2 = ConvCapsuleLayer(kernel_size=5, NP=self.NP2, LC=self.LP1, LP=self.LP2, strides=self.stride2, padding='same', routings=3, deconv=False)
        self.cap_3 = ConvCapsuleLayer(kernel_size=5, NP=self.NP3, LC=self.LP2, LP=self.LP3, strides=self.stride3, padding='same', routings=3, deconv=False)
        self.cap_4 = ConvCapsuleLayer(kernel_size=5, NP=self.NP4, LC=self.LP3, LP=self.LP4, strides=self.stride4, padding='same', routings=3, deconv=False)
        self.cap_5 = ConvCapsuleLayer(kernel_size=5, NP=self.NP5, LC=self.LP4, LP=self.LP5, strides=self.stride5, padding='same', routings=3, deconv=False)
        self.cap_6 = ConvCapsuleLayer(kernel_size=5, NP=self.NP6, LC=self.LP5, LP=self.LP6, strides=self.stride6, padding='same', routings=3, deconv=False)
        self.up1   = ConvCapsuleLayer(kernel_size=4, NP=self.NU1, LC=self.LP6, LP=self.LU1, strides=self.stridu1, padding='same', routings=3, deconv=True)
        self.cap_7 = ConvCapsuleLayer(kernel_size=5, NP=self.NP7, LC=self.LP4, LP=self.LP7, strides=self.stride7, padding='same', routings=3, deconv=False)
        self.up2   = ConvCapsuleLayer(kernel_size=4, NP=self.NU2, LC=self.LP3, LP=self.LU2, strides=self.stridu2, padding='same', routings=3, deconv=True)
        self.cap_8 = ConvCapsuleLayer(kernel_size=5, NP=self.NP8, LC=self.LP2, LP=self.LP8, strides=self.stride8, padding='same', routings=3, deconv=False)
        self.up3   = ConvCapsuleLayer(kernel_size=4, NP=self.NU3, LC=self.LP8, LP=self.LU3, strides=self.stridu3, padding='same', routings=3, deconv=True)
        self.cap_9 = ConvCapsuleLayer(kernel_size=1, NP=self.NP9, LC=self.LP8, LP=self.LP9, strides=self.stride9, padding='same', routings=3, deconv=False)

        # Layer 0: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.NP0 * self.LP0, kernel_size=5, padding=5//2, stride=self.stride0)
        self.bn1 = nn.BatchNorm2d(self.NP0 * self.LP0)

    def forward(self, x):
        B, _, H, W = x.shape
        x = F.relu(self.bn1(self.conv1(x)))  # 512
        x0 = x.view(B, self.NP0, self.LP0, H, W)
        x1 = self.cap_1(x0)  # 256
        x2 = self.cap_2(x1)  # 256
        x3 = self.cap_3(x2)  # 128
        x4 = self.cap_4(x3)  # 128
        x5 = self.cap_5(x4)  # 64
        x6 = self.cap_6(x5)  # 64
        
        u1 = self.up1(x6)    # 128
        c1 = torch.cat([x4, u1], dim=1)
        x7 = self.cap_7(c1)  # 128
        c2 = torch.cat([x7, x3], dim=1)
        u2 = self.up2(c2)     # 256
        x8 = self.cap_8(u2)  # 256
        c3 = torch.cat([x8, x2], dim=1)
        u3 = self.up3(c3)     # 512
        x9 = self.cap_9(u3)  # 512
        length = torch.norm(x9, dim=-3).squeeze(dim=1)
        return length
