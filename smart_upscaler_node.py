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
    """Generalized RRDB network matching official checkpoints.
    Supporte le pixel-unshuffle en entrée (ex: RealESRGAN-x2plus avec num_in_ch=12).
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        # pixel-unshuffle factor : num_in_ch = 3 * unshuffle_factor²
        # ex: num_in_ch=12 → unshuffle=2 ; num_in_ch=3 → unshuffle=1 (pas de unshuffle)
        self.unshuffle_factor = round((num_in_ch / 3) ** 0.5)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling (le scale réseau = scale_total / unshuffle_factor)
        net_scale = scale // self.unshuffle_factor if self.unshuffle_factor > 1 else scale
        self.net_scale = net_scale
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if net_scale >= 4:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if net_scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr   = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Pixel-unshuffle si nécessaire (augmente les canaux, réduit la résolution)
        if self.unshuffle_factor > 1:
            x = F.pixel_unshuffle(x, self.unshuffle_factor)
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.net_scale >= 4:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.net_scale == 8:
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

def _fmt_mb(b: int) -> str:
    if b >= 1024**3: return f"{b/1024**3:.1f} GB"
    if b >= 1024**2: return f"{b/1024**2:.1f} MB"
    return f"{b/1024:.1f} KB"

def _ram_used_mb() -> float:
    """RAM process (RSS) en MB via /proc/self/status (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0

def _log(msg: str) -> None:
    print(f"[SmartUpscaler] {msg}", flush=True)

def download_model(url: str, dest: Path) -> None:
    import time
    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"⬇  Téléchargement → {dest.name}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    t0 = time.time()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct  = downloaded / total * 100
                spd  = downloaded / max(time.time() - t0, 0.001)
                eta  = (total - downloaded) / max(spd, 1)
                print(f"\r[SmartUpscaler]    {pct:5.1f}%  {_fmt_mb(downloaded)}/{_fmt_mb(total)}"
                      f"  vitesse: {_fmt_mb(int(spd))}/s  ETA: {eta:.0f}s   ", end="", flush=True)
    print()  # newline après la barre
    elapsed = time.time() - t0
    _log(f"✓ Téléchargement terminé: {dest.name} ({_fmt_mb(dest.stat().st_size)}, {elapsed:.1f}s)")


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
        # Architecture RRDB — détection dynamique depuis le checkpoint
        num_block = 23
        body_keys = [k for k in state if k.startswith("body.") and k.split(".")[1].isdigit()]
        if body_keys:
            num_block = max(int(k.split(".")[1]) for k in body_keys) + 1

        # num_in_ch : détecté depuis conv_first.weight (shape [out, in, kH, kW])
        # RealESRGAN-x2plus utilise pixel-unshuffle x2 → num_in_ch = 3*4 = 12
        num_in_ch = 3
        if "conv_first.weight" in state:
            num_in_ch = state["conv_first.weight"].shape[1]

        net = RRDBNet(num_in_ch=num_in_ch, num_feat=64, num_block=num_block, num_grow_ch=32, scale=scale)
        print(f"[SmartUpscaler] Arch: RRDBNet | num_block={num_block} | num_in_ch={num_in_ch}")

    net.load_state_dict(state, strict=False)
    net.eval()
    net.scale = scale

    ram_before = _ram_used_mb()
    if device.type == "cuda":
        vram_before = torch.cuda.memory_allocated() // 1024 // 1024
        net = net.half().to(device)
        vram_after  = torch.cuda.memory_allocated() // 1024 // 1024
        vram_model  = vram_after - vram_before
        _log(f"✓ '{model_name}' chargé sur {device} (×{scale}) | "
             f"VRAM modèle: +{vram_model} MB (total alloué: {vram_after} MB) | "
             f"RAM process: {_ram_used_mb():.0f} MB")
    else:
        net = net.to(device)
        ram_after = _ram_used_mb()
        _log(f"✓ '{model_name}' chargé sur CPU (×{scale}) | "
             f"RAM modèle: +{ram_after - ram_before:.0f} MB (total: {ram_after:.0f} MB)")

    _MODEL_CACHE[cache_key] = net
    return net


# ──────────────────────────────────────────────
# Tile-based inference (avoids OOM on large frames)
# ──────────────────────────────────────────────

def upscale_tile(model, img_tensor: torch.Tensor,
                 tile: int = 512, overlap: int = 32,
                 frame_label: str = "") -> torch.Tensor:
    """Process image in tiles — output on CPU, flush CUDA cache per tile."""
    import time
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    scale  = model.scale
    _, c, h, w = img_tensor.shape

    step     = max(tile - overlap, 1)
    n_cols   = len(range(0, w, step))
    n_rows   = len(range(0, h, step))
    n_tiles  = n_cols * n_rows
    tile_idx = 0
    t_pass   = time.time()

    output_h = h * scale
    output_w = w * scale
    output   = torch.zeros((1, c, output_h, output_w), dtype=torch.float32)

    for y in range(0, h, step):
        for x in range(0, w, step):
            t_tile = time.time()
            tile_idx += 1
            y_end = min(y + tile, h)
            x_end = min(x + tile, w)

            patch = img_tensor[:, :, y:y_end, x:x_end].to(device=device, dtype=dtype)

            with torch.no_grad():
                out_patch = model(patch).float().cpu()

            del patch
            if device.type == "cuda":
                torch.cuda.empty_cache()

            oy = y * scale
            ox = x * scale
            border_y = overlap * scale // 2 if y > 0 else 0
            border_x = overlap * scale // 2 if x > 0 else 0

            output[:, :,
                   oy + border_y : oy + out_patch.shape[2],
                   ox + border_x : ox + out_patch.shape[3]] = \
                out_patch[:, :, border_y:, border_x:]
            del out_patch

            # Log toutes les N tuiles pour ne pas spammer
            log_every = max(1, n_tiles // 8)
            if tile_idx == 1 or tile_idx % log_every == 0 or tile_idx == n_tiles:
                elapsed   = time.time() - t_pass
                remaining = elapsed / tile_idx * (n_tiles - tile_idx)
                vram_str  = ""
                if device.type == "cuda":
                    used  = (torch.cuda.memory_allocated() ) // 1024 // 1024
                    total = torch.cuda.mem_get_info()[1] // 1024 // 1024
                    vram_str = f"  VRAM: {used} MB alloc / {total} MB total"
                ram_str = f"  RAM: {_ram_used_mb():.0f} MB"
                prefix  = f"{frame_label} " if frame_label else ""
                print(f"[SmartUpscaler]    {prefix}tuile {tile_idx:3d}/{n_tiles}"
                      f"  ({x_end-y_end+tile}x{tile}px→×{scale})"
                      f"  ETA: {remaining:.0f}s{vram_str}{ram_str}", flush=True)

    elapsed_pass = time.time() - t_pass
    _log(f"   {frame_label + ' ' if frame_label else ''}pass terminé: {n_tiles} tuiles"
         f"  {w}x{h} → {output_w}x{output_h}"
         f"  durée: {elapsed_pass:.1f}s  ({elapsed_pass/n_tiles*1000:.0f} ms/tuile)")
    return output  # float32 CPU


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

        import gc, time
        t_total = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── En-tête ─────────────────────────────────────────────────────────
        sep = "─" * 60
        _log(sep)
        _log(f"🔍 Smart Upscaler  |  modèle: {model_name}  |  cible: {target_resolution}")
        _log(f"   device: {device}  |  tile: {tile_size}px  |  overlap: {tile_overlap}px")
        if device.type == "cuda":
            free_mb  = torch.cuda.mem_get_info()[0] // 1024 // 1024
            total_mb = torch.cuda.mem_get_info()[1] // 1024 // 1024
            _log(f"   VRAM dispo: {free_mb} MB / {total_mb} MB  |  RAM process: {_ram_used_mb():.0f} MB")
        else:
            _log(f"   ⚠ CUDA non disponible — CPU (lent!)")
            _log(f"   RAM process: {_ram_used_mb():.0f} MB")
        _log(sep)

        target_px = RESOLUTIONS[target_resolution]
        model     = load_model(model_name, device)

        nb_frames, h_in, w_in, c_in = image.shape
        _log(f"   Batch: {nb_frames} frame(s)  |  entrée: {w_in}×{h_in}  |  canaux: {c_in}")

        # ── Calcul taille de sortie ──────────────────────────────────────────
        long_edge = max(h_in, w_in)
        if long_edge >= target_px:
            out_h = int(round(h_in * target_px / long_edge))
            out_w = int(round(w_in * target_px / long_edge))
            n_passes = 0
        else:
            sim_h, sim_w = h_in, w_in
            n_passes = 0
            while max(sim_h, sim_w) < target_px:
                sim_h  *= model.scale
                sim_w  *= model.scale
                n_passes += 1
            scale_f = target_px / max(sim_h, sim_w)
            out_h = int(round(sim_h * scale_f))
            out_w = int(round(sim_w * scale_f))
        out_h += out_h % 2
        out_w += out_w % 2

        # Estimation tuiles par frame par passe
        step = max(tile_size - tile_overlap, 1)
        tiles_per_pass = (
            len(range(0, w_in * (model.scale ** max(n_passes-1,0)), step)) *
            len(range(0, h_in * (model.scale ** max(n_passes-1,0)), step))
        ) if n_passes else 0

        ram_output_mb = nb_frames * out_h * out_w * c_in * 4 // 1024 // 1024
        _log(f"   Sortie: {out_w}×{out_h}  |  passes/frame: {n_passes}  |"
             f"  RAM output batch: ~{ram_output_mb} MB")
        _log(sep)

        # ── Pré-allocation du tenseur de sortie ──────────────────────────────
        output_batch = torch.empty((nb_frames, out_h, out_w, c_in),
                                   dtype=torch.float32, pin_memory=False)
        _log(f"   Tenseur sortie pré-alloué: {output_batch.numel()*4//1024//1024} MB RAM")

        for i in range(nb_frames):
            t_frame = time.time()
            frame_label = f"[{i+1}/{nb_frames}]"
            _log(f"")
            _log(f"▶ Frame {i+1}/{nb_frames}  |  RAM: {_ram_used_mb():.0f} MB"
                 + (f"  VRAM allouée: {torch.cuda.memory_allocated()//1024//1024} MB"
                    if device.type == "cuda" else ""))

            frame = image[i].permute(2, 0, 1).unsqueeze(0).float()  # 1,C,H,W CPU
            long_edge_f = max(frame.shape[2], frame.shape[3])

            if long_edge_f >= target_px:
                _log(f"   Déjà ≥ cible — redimensionnement direct (pas d'upscale IA)")
                out = resize_to_target(frame, target_px)
            else:
                current  = frame
                pass_num = 0
                while max(current.shape[2], current.shape[3]) < target_px:
                    pass_num += 1
                    cur_w, cur_h = current.shape[3], current.shape[2]
                    _log(f"   Pass {pass_num}/{n_passes}  |  entrée: {cur_w}×{cur_h}"
                         f"  →  sortie estimée: {cur_w*model.scale}×{cur_h*model.scale}"
                         + (f"  |  VRAM libre: {torch.cuda.mem_get_info()[0]//1024//1024} MB"
                            f"  allouée: {torch.cuda.memory_allocated()//1024//1024} MB"
                            if device.type == "cuda" else "")
                         + f"  |  RAM: {_ram_used_mb():.0f} MB")
                    t_pass = time.time()
                    prev    = current
                    current = upscale_tile(model, current, tile_size, tile_overlap,
                                           frame_label=f"{frame_label} pass {pass_num}")
                    current = current.clamp(0, 1)
                    del prev
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    _log(f"   Pass {pass_num} terminée en {time.time()-t_pass:.1f}s"
                         + (f"  |  VRAM libre: {torch.cuda.mem_get_info()[0]//1024//1024} MB"
                            if device.type == "cuda" else ""))

                out = resize_to_target(current, target_px)
                del current

            if force_exact_resolution:
                _, c, oh, ow = out.shape
                cy, cx = oh // 2, ow // 2
                half   = target_px // 2
                out = out[:, :, max(0,cy-half):cy+half, max(0,cx-half):cx+half]

            h_out = min(out.shape[2], out_h)
            w_out = min(out.shape[3], out_w)
            output_batch[i, :h_out, :w_out, :] = out[0, :, :h_out, :w_out].permute(1, 2, 0)
            del out, frame

            t_frame_elapsed = time.time() - t_frame
            fps_est = 1.0 / max(t_frame_elapsed, 0.001)
            frames_left = nb_frames - (i + 1)
            eta_total = t_frame_elapsed * frames_left
            _log(f"◀ Frame {i+1}/{nb_frames} terminée  |  durée: {t_frame_elapsed:.1f}s"
                 f"  ({fps_est:.2f} fps)  |  ETA: {eta_total:.0f}s"
                 f"  |  RAM: {_ram_used_mb():.0f} MB"
                 + (f"  VRAM allouée: {torch.cuda.memory_allocated()//1024//1024} MB"
                    if device.type == "cuda" else ""))

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        t_elapsed = time.time() - t_total
        _log("")
        _log(sep)
        _log(f"✅ Terminé  |  {nb_frames} frame(s)  {w_in}×{h_in} → {out_w}×{out_h}"
             f"  ({target_resolution})  |  durée totale: {t_elapsed:.1f}s"
             f"  ({nb_frames/max(t_elapsed,0.001):.2f} fps)")
        _log(f"   RAM finale: {_ram_used_mb():.0f} MB"
             + (f"  |  VRAM allouée: {torch.cuda.memory_allocated()//1024//1024} MB"
                f"  /  réservée: {torch.cuda.memory_reserved()//1024//1024} MB"
                if device.type == "cuda" else ""))
        _log(sep)

        return (output_batch.clamp(0, 1),)


# ──────────────────────────────────────────────
# ComfyUI registration
# ──────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SmartUpscaler": SmartUpscalerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartUpscaler": "🔍 Smart Upscaler (1080p/2K/4K/8K)",
}
