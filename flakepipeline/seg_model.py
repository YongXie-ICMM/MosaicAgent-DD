"""U-Net definition, shaped to match model_0409_all.pth exactly.

The original predict_tongji_sum.py did `from src import UNet`, but src/ was never
handed over, so nobody could actually run that code. Here the network is
reconstructed from the tensor shapes found in the checkpoint itself:
load_state_dict(strict=True) matches every key, and no external private module is
needed any more.

The shapes that matter: in_channels=3, num_classes=4, bilinear=True, base_c=32.
load_model_once() in predict_tongji_sum.py called UNet(num_classes=...) without
passing in_channels or base_c, relying on locally edited defaults in some machine's
copy of src/unet.py.
"""
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        mid_ch = out_ch if mid_ch is None else mid_ch
        super().__init__(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class Down(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.MaxPool2d(2, stride=2), DoubleConv(in_ch, out_ch))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size()[2] - x1.size()[2]
        dx = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Sequential):
    def __init__(self, in_ch, num_classes):
        super().__init__(nn.Conv2d(in_ch, num_classes, kernel_size=1))


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, bilinear=True, base_c=32):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return {"out": self.out_conv(x)}


def _load_checkpoint(weights_path):
    """Keep legacy checkpoint loading explicit where PyTorch supports the option."""
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(weights_path, **kwargs)


def load_segmenter(weights_path, device="cpu"):
    """Load from a checkpoint. Shapes are cross-checked; a mismatch raises rather
    than silently falling back to a different architecture."""
    ck = _load_checkpoint(weights_path)
    sd = ck["model"] if "model" in ck else ck
    sd = {k: v for k, v in sd.items() if "aux" not in k}
    base_c = sd["in_conv.0.weight"].shape[0]
    in_ch = sd["in_conv.0.weight"].shape[1]
    n_cls = sd["out_conv.0.weight"].shape[0]
    model = UNet(in_channels=in_ch, num_classes=n_cls, bilinear=True, base_c=base_c)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    meta = {"base_c": int(base_c), "in_channels": int(in_ch), "num_classes": int(n_cls),
            "epoch": int(ck.get("epoch", -1)) if isinstance(ck, dict) else -1}
    return model, meta
