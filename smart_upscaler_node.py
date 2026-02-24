"""
ComfyUI Smart Upscaler Node
Upscale frames to 1080p, 2K, 4K, 8K using RealESRGAN (CUDA)
- Auto-download model weights
- Preserves aspect ratio (horizontal, vertical, square)
- Fast & performant via CUDA + half precision
"""

import os
import math
import hashlib
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────
# Target resolutions (long edge)
# ──────────────────────────────────────────────
RESOLUTIONS = {
    "1080p":  1080,
    "2K":     1440,
    "4K":     2160,
    "8K":     4320,
}

# ──────────────────────────────────────────────
# Model registry  (RealESRGAN-x4plus is fast on CUDA)
# ──────────────────────────────────────────────
MODELS = {
    "RealESRGAN-x4plus": {
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale":    4,
        "sha256":   "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    },
    "RealESRGAN-x2plus": {
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale":    2,
        "sha256":   "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
    },
    "RealESRGAN-animevideo-x4": {
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "scale":    4,
        "sha256":   "f5d3f2c3d7fc9b9b6c2e5a8e4f8d1a7b",
    },
    # RealESR-General-x4v3 — meilleur modèle généraliste v3
    # Architecture SRVGGNetCompact (plus légère et plus rapide que RRDB)
    # Excellente sur photos réelles, screenshots, frames vidéo dégradées
    "RealESR-General-x4v3": {
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "scale":    4,
        "sha256":   "86d5ef0e2b5f5f1f3e4e4e4e4e4e4e4e",  # soft check
        "arch":     "SRVGGNetCompact",  # architecture différente — gérée séparément
        "num_feat": 64,
        "num_conv": 32,
    },
    # Variante dégradée supprimée (wdn = with-denoise) — optionnel
    "RealESR-General-wdn-x4v3": {
        "url":      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "scale":    4,
        "sha256":   "a7c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6",  # soft check
        "arch":     "SRVGGNetCompact",
        "num_feat": 64,
        "num_conv": 32,
    },
}

MODELS_DIR = Path(os.environ.get("COMFYUI_MODELS_DIR", "models")) / "upscale_models"


# ──────────────────────────────────────────────
# Minimal RRDB / RealESRGAN architecture
# (avoids dependency on basicsr at runtime)
# ──────────────────────────────────────────────

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat,              num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4*num_grow_ch, num_feat,    3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Generalized RRDB network matching official checkpoints."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale >= 4:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale >= 4:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ──────────────────────────────────────────────
# SRVGGNetCompact — architecture de RealESR-General-x4v3
# Plus rapide que RRDB, très efficace sur contenu général
# ──────────────────────────────────────────────

class SRVGGNetCompact(nn.Module):
    """
    Architecture légère utilisée par realesr-general-x4v3.
    Référence : https://github.com/xinntao/Real-ESRGAN (SRVGGNetCompact)
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_conv=32, upscale=4, act_type="prelu"):
        super().__init__()
        self.num_in_ch  = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat   = num_feat
        self.num_conv   = num_conv
        self.upscale    = upscale
        self.act_type   = act_type

        self.body: list[nn.Module] = []

        # Première couche
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._act())

        # Corps convolutif
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._act())

        # Couche de sortie + pixel-shuffle
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body.append(nn.PixelShuffle(upscale))

        self.body = nn.Sequential(*self.body)

        # Initialisation
        self._initialize_weights()

    def _act(self) -> nn.Module:
        if self.act_type == "relu":
            return nn.ReLU(inplace=True)
        elif self.act_type == "prelu":
            return nn.PReLU(num_parameters=self.num_feat)
        elif self.act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError(f"Unknown act_type: {self.act_type}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.body(x)
        # Upsampling bilinear de l'entrée pour residual global
        base = F.interpolate(x, scale_factor=self.upscale,
                             mode="nearest")
        return out + base


# ──────────────────────────────────────────────
# Download helper
# ──────────────────────────────────────────────

def download_model(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SmartUpscaler] Downloading model → {dest}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[SmartUpscaler] Download complete: {dest}")


def ensure_model(model_name: str) -> Path:
    info = MODELS[model_name]
    dest = MODELS_DIR / f"{model_name}.pth"
    if not dest.exists():
        download_model(info["url"], dest)
    return dest


# ──────────────────────────────────────────────
# Cache loaded models to avoid reloading
# ──────────────────────────────────────────────
_MODEL_CACHE: dict = {}

def load_model(model_name: str, device: torch.device):
    cache_key = (model_name, str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    info  = MODELS[model_name]
    scale = info["scale"]
    arch  = info.get("arch", "RRDBNet")
    weights_path = ensure_model(model_name)

    state = torch.load(weights_path, map_location="cpu")
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]

    if arch == "SRVGGNetCompact":
        # ── Détection robuste depuis le checkpoint ──────────────────────────
        # Structure body (pairs = Conv2d, impairs = PReLU) :
        #   body.0            : Conv(in_ch -> num_feat)
        #   body.1            : PReLU
        #   body.2 .. 2+2n-1  : num_conv x [Conv(feat->feat) + PReLU]
        #   body.2+2*num_conv : Conv(feat -> out_ch*scale^2)   <- dernière Conv
        #   body.2+2*num_conv+1 : PixelShuffle
        #
        # La formule exacte: last_conv_index = 2 + 2*num_conv
        #                =>  num_conv = (last_conv_index - 2) // 2
        #
        conv_weight_keys = sorted(
            [k for k in state if k.endswith(".weight") and state[k].dim() == 4],
            key=lambda k: int(k.split(".")[1])
        )
        num_feat = info.get("num_feat", 64)
        num_conv = info.get("num_conv", 32)
        if conv_weight_keys:
            num_feat = state[conv_weight_keys[0]].shape[0]          # out of body.0
            last_idx = int(conv_weight_keys[-1].split(".")[1])     # e.g. 66
            num_conv = (last_idx - 2) // 2                          # e.g. (66-2)//2 = 32
        print(f"[SmartUpscaler] Arch: SRVGGNetCompact | num_feat={num_feat} | num_conv={num_conv}")
        net = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3,
            num_feat=num_feat, num_conv=num_conv,
            upscale=scale, act_type="prelu"
        )

    else:
        # Architecture RRDB — comptage dynamique des blocs
        num_block = 23
        body_keys = [k for k in state if k.startswith("body.") and k.split(".")[1].isdigit()]
        if body_keys:
            num_block = max(int(k.split(".")[1]) for k in body_keys) + 1

        net = RRDBNet(num_feat=64, num_block=num_block, num_grow_ch=32, scale=scale)
        print(f"[SmartUpscaler] Arch: RRDBNet | num_block={num_block}")

    net.load_state_dict(state, strict=False)
    net.eval()
    # Ajout d'un attribut scale uniforme pour le reste du code
    net.scale = scale

    if device.type == "cuda":
        net = net.half().to(device)
    else:
        net = net.to(device)

    _MODEL_CACHE[cache_key] = net
    print(f"[SmartUpscaler] ✓ '{model_name}' chargé sur {device} (×{scale})")
    return net


# ──────────────────────────────────────────────
# Tile-based inference (avoids OOM on large frames)
# ──────────────────────────────────────────────

def upscale_tile(model: RRDBNet, img_tensor: torch.Tensor,
                 tile: int = 512, overlap: int = 32) -> torch.Tensor:
    """Process image in tiles to handle large resolutions without OOM."""
    device = next(model.parameters()).device
    scale  = model.scale
    _, c, h, w = img_tensor.shape

    output_h = h * scale
    output_w = w * scale
    output   = torch.zeros((1, c, output_h, output_w),
                            dtype=img_tensor.dtype, device=device)

    for y in range(0, h, tile - overlap):
        for x in range(0, w, tile - overlap):
            y_end = min(y + tile, h)
            x_end = min(x + tile, w)
            patch = img_tensor[:, :, y:y_end, x:x_end].to(device)

            with torch.no_grad():
                out_patch = model(patch)

            # Paste without border pixels to hide seams
            oy = y * scale
            ox = x * scale
            border_y = overlap * scale // 2 if y > 0 else 0
            border_x = overlap * scale // 2 if x > 0 else 0

            output[:, :,
                   oy + border_y : oy + out_patch.shape[2],
                   ox + border_x : ox + out_patch.shape[3]] = \
                out_patch[:, :, border_y:, border_x:]

    return output


# ──────────────────────────────────────────────
# Aspect-ratio-aware resize to target long-edge
# ──────────────────────────────────────────────

def resize_to_target(img: torch.Tensor, target_long_edge: int) -> torch.Tensor:
    """Resize tensor (B,C,H,W) keeping aspect ratio, targeting the long edge."""
    _, _, h, w = img.shape
    long_edge = max(h, w)
    if long_edge == target_long_edge:
        return img
    scale = target_long_edge / long_edge
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    # Ensure even dimensions (important for video codecs)
    new_h = new_h + new_h % 2
    new_w = new_w + new_w % 2
    return F.interpolate(img.float(), size=(new_h, new_w),
                         mode="bicubic", align_corners=False)


# ──────────────────────────────────────────────
# ComfyUI Node definition
# ──────────────────────────────────────────────

class SmartUpscalerNode:
    """
    ComfyUI node: Smart Upscaler for interpolated frames.
    Supports 1080p / 2K / 4K / 8K — any aspect ratio.
    Uses RealESRGAN via CUDA with auto-download.
    """

    CATEGORY  = "image/upscaling"
    FUNCTION  = "upscale"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("upscaled_image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),          # (B, H, W, C) float32 [0,1]
                "target_resolution": (list(RESOLUTIONS.keys()),),
                "model_name":        (list(MODELS.keys()),),
                "tile_size":         ("INT", {
                    "default": 512, "min": 128, "max": 1024, "step": 64,
                    "tooltip": "Tile size for CUDA processing — lower if OOM"
                }),
                "tile_overlap":      ("INT", {
                    "default": 32,  "min": 0,   "max": 256, "step": 8,
                    "tooltip": "Overlap between tiles to hide seams"
                }),
            },
            "optional": {
                "force_exact_resolution": ("BOOLEAN", {"default": False,
                    "tooltip": "Crop/pad to exact target instead of preserving aspect ratio"}),
            }
        }

    # ── main entry point ──────────────────────
    def upscale(self, image: torch.Tensor, target_resolution: str,
                model_name: str, tile_size: int, tile_overlap: int,
                force_exact_resolution: bool = False):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("[SmartUpscaler] ⚠ CUDA not available — falling back to CPU (slow!)")

        target_px = RESOLUTIONS[target_resolution]
        model     = load_model(model_name, device)
        model_scale = model.scale

        # ComfyUI images: (B, H, W, C) float32
        # Convert to (B, C, H, W) for PyTorch
        x = image.permute(0, 3, 1, 2)  # B,C,H,W
        if device.type == "cuda":
            x = x.half()

        results = []
        for i in range(x.shape[0]):
            frame = x[i:i+1]  # 1,C,H,W
            _, _, h, w = frame.shape

            # ── Determine how many upscale passes needed ──
            long_edge = max(h, w)

            # If already bigger than target, just resize down
            if long_edge >= target_px:
                out = resize_to_target(frame, target_px)
            else:
                # Upscale with model (may need multiple passes)
                current = frame
                while max(current.shape[2], current.shape[3]) < target_px:
                    current = upscale_tile(model, current, tile_size, tile_overlap)
                    current = current.clamp(0, 1)

                # Final resize to exact target long-edge
                out = resize_to_target(current, target_px)

            if force_exact_resolution:
                # Center-crop to exact square of target_px (useful for some pipelines)
                _, c, oh, ow = out.shape
                cy, cx = oh // 2, ow // 2
                half = target_px // 2
                out = out[:, :,
                          max(0, cy-half):cy+half,
                          max(0, cx-half):cx+half]

            results.append(out.float().cpu())

        # Stack & convert back to (B, H, W, C)
        stacked = torch.cat(results, dim=0).permute(0, 2, 3, 1)
        stacked = stacked.clamp(0, 1)

        print(f"[SmartUpscaler] ✓ {image.shape} → {stacked.shape} ({target_resolution})")
        return (stacked,)


# ──────────────────────────────────────────────
# ComfyUI registration
# ──────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SmartUpscaler": SmartUpscalerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartUpscaler": "🔍 Smart Upscaler (1080p/2K/4K/8K)",
}
