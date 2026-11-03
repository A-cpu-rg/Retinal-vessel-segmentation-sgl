"""CON model from the SGL paper: EnhanceNet followed by SegNet (two U-Nets).

Follows the official code (SHI-Labs/SGL-Retinal-Vessel-Segmentation, model/con.py).
Only the SegNet output is compared with labels; EnhanceNet learns through it.
"""
import torch
import torch.nn as nn

from config import SIZE_MULTIPLE


def _conv_block(channel_in, channel_out, first=False):
    # The first block of each U-Net has no PReLU/BatchNorm before its first conv.
    if first:
        return nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 3, 1, 1),
            nn.PReLU(),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
    return nn.Sequential(
        nn.PReLU(),
        nn.BatchNorm2d(channel_in),
        nn.Conv2d(channel_in, channel_out, 3, 1, 1),
        nn.PReLU(),
        nn.BatchNorm2d(channel_out),
        nn.Conv2d(channel_out, channel_out, 3, 1, 1),
    )


def _upconv(channel_in, channel_out):
    return nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)


class _UNet(nn.Module):
    """5-level U-Net (32 to 512 channels). The last block also gets the U-Net's input."""

    def __init__(self, in_ch, out_ch, scale_input):
        super().__init__()
        self.scale_input = scale_input  # EnhanceNet gets RGB 0..255 and divides by 255
        self.conv1 = _conv_block(in_ch, 32, first=True)
        self.conv2 = _conv_block(32, 64)
        self.conv3 = _conv_block(64, 128)
        self.conv4 = _conv_block(128, 128 * 2)
        self.conv5 = _conv_block(128 * 2, 128 * 4)
        self.pool = nn.MaxPool2d(2)
        self.upconv1 = _upconv(64, 32)
        self.upconv2 = _upconv(128, 64)
        self.upconv3 = _upconv(128 * 2, 128)
        self.upconv4 = _upconv(128 * 4, 128 * 2)
        self.conv6 = _conv_block(128 * 4, 128 * 2)
        self.conv7 = _conv_block(128 * 2, 128)
        self.conv8 = _conv_block(128, 64)
        self.conv9 = _conv_block(64, 32)
        self.conv11 = _conv_block(32 + in_ch, out_ch)

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % SIZE_MULTIPLE or w % SIZE_MULTIPLE:
            raise ValueError(f"Input size {h}x{w} must be a multiple of {SIZE_MULTIPLE}. "
                             "Pad the image first.")
        if self.scale_input:
            x = x / 255.0
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))
        u4 = self.conv6(torch.cat([self.upconv4(x5), x4], 1))
        u3 = self.conv7(torch.cat([self.upconv3(u4), x3], 1))
        u2 = self.conv8(torch.cat([self.upconv2(u3), x2], 1))
        u1 = self.conv9(torch.cat([self.upconv1(u2), x1], 1))
        u1 = torch.cat([u1, x], 1)
        return torch.sigmoid(self.conv11(u1))


class EnhanceNet(_UNet):
    def __init__(self):
        super().__init__(in_ch=3, out_ch=3, scale_input=True)


class SegNet(_UNet):
    def __init__(self):
        super().__init__(in_ch=3, out_ch=1, scale_input=False)


class CON(nn.Module):
    """Input: (B, 3, H, W) RGB, values 0..255.
    Output: (enhanced image (B, 3, H, W), vessel probability (B, 1, H, W))."""

    def __init__(self):
        super().__init__()
        self.s1 = EnhanceNet()
        self.s2 = SegNet()

    def forward(self, x):
        enhance = self.s1(x)
        seg = self.s2(enhance)
        return enhance, seg


def build_model():
    return CON()
