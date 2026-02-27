"""
ComfyUI-AnimeUpscale4K
━━━━━━━━━━━━━━━━━━━━━━
Custom node pack for post-processing anime-style videos generated with Wan2.1/2.2.
Includes: Real-ESRGAN upscaling (4K), color correction, sharpening, temporal denoising,
           and video export with FFmpeg.

All models are auto-downloaded on first use.

⚠️ ZERO external AI dependencies — RRDBNet architecture and Real-ESRGAN inference
   are fully embedded. No basicsr, no realesrgan, no gfpgan needed.
   Only requires: torch, numpy, opencv, requests.
"""

import hashlib
import math
import os
import subprocess
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths  # ComfyUI utility


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Embedded RRDBNet Architecture
# (Source: xinntao/basicsr, MIT License — embedded to avoid dependency)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
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
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Embedded Real-ESRGAN Tiled Inference
# (Source: xinntao/Real-ESRGAN, BSD License — embedded to avoid dependency)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RealESRGANInfer:
    """Minimal Real-ESRGAN inference with tiled processing. No external deps."""

    def __init__(self, scale, model_path, model, tile=256, tile_pad=10, pre_pad=10,
                 half=False, device='cuda'):
        self.scale = scale
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half
        self.device = torch.device(device)

        loadnet = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        elif 'params' in loadnet:
            keyname = 'params'
        else:
            keyname = None

        if keyname:
            model.load_state_dict(loadnet[keyname], strict=True)
        else:
            model.load_state_dict(loadnet, strict=True)

        model.eval()
        model = model.to(self.device)
        if self.half:
            model = model.half()
        self.model = model

    def enhance(self, img_bgr, outscale=4):
        """Enhance a BGR uint8 image. Returns: (output_bgr_uint8, None)"""
        img = img_bgr.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)
        img = img.to(self.device)
        if self.half:
            img = img.half()

        if self.pre_pad > 0:
            img = F.pad(img, (self.pre_pad,) * 4, 'reflect')

        if self.tile > 0:
            output = self._tile_process(img)
        else:
            with torch.no_grad():
                output = self.model(img)

        if self.pre_pad > 0:
            pp = self.pre_pad * self.scale
            output = output[:, :, pp:-pp, pp:-pp]

        output = output.squeeze(0).float().cpu().clamp(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (output * 255.0).round().astype(np.uint8)

        h, w = img_bgr.shape[:2]
        target_h, target_w = int(h * outscale), int(w * outscale)
        oh, ow = output.shape[:2]
        if oh != target_h or ow != target_w:
            interp = cv2.INTER_AREA if oh > target_h else cv2.INTER_LANCZOS4
            output = cv2.resize(output, (target_w, target_h), interpolation=interp)

        return output, None

    def _tile_process(self, img):
        """Process image in tiles to save VRAM."""
        batch, channel, height, width = img.shape
        output_h = height * self.scale
        output_w = width * self.scale
        output = img.new_zeros((batch, channel, output_h, output_w))

        tiles_x = math.ceil(width / self.tile)
        tiles_y = math.ceil(height / self.tile)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.tile
                ofs_y = y * self.tile
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile, height)

                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                input_tile = img[:, :, input_start_y_pad:input_end_y_pad,
                                 input_start_x_pad:input_end_x_pad]

                with torch.no_grad():
                    output_tile = self.model(input_tile)

                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * self.scale

                output[:, :, output_start_y:output_end_y,
                       output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                output_start_x_tile:output_end_x_tile]

        return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY = "🎌 Anime Upscale 4K"
MODEL_DIR = os.path.join(folder_paths.models_dir, "anime_upscale")
TARGET_4K = (3840, 2160)

MODELS_REGISTRY = {
    "realesr-animevideov3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "filename": "realesr-animevideov3.pth",
        "num_block": 6,
        "num_grow_ch": 32,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "num_block": 6,
        "num_grow_ch": 32,
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model downloader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ensure_model(model_name: str) -> str:
    """Download model if not present, return path."""
    import requests

    info = MODELS_REGISTRY[model_name]
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, info["filename"])

    if os.path.exists(model_path):
        print(f"[AnimeUpscale] ✅ Modèle en cache: {model_path}")
        return model_path

    url = info["url"]
    print(f"[AnimeUpscale] ⬇️  Téléchargement {model_name} depuis {url}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    tmp_path = model_path + ".tmp"
    downloaded = 0
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r[AnimeUpscale]    {pct:.1f}% ({downloaded // 1024}KB / {total // 1024}KB)",
                      end="", flush=True)
    print()

    os.rename(tmp_path, model_path)
    print(f"[AnimeUpscale] ✅ Modèle sauvegardé: {model_path}")
    return model_path


def load_realesrgan(model_name: str, tile: int, fp16: bool, device: str):
    """Load Real-ESRGAN upsampler using embedded architecture."""
    model_path = ensure_model(model_name)
    info = MODELS_REGISTRY[model_name]

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=info["num_block"],
        num_grow_ch=info["num_grow_ch"],
        scale=4,
    )

    upsampler = RealESRGANInfer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=10,
        half=fp16 and device == "cuda",
        device=device,
    )
    return upsampler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ComfyUI tensor ↔ OpenCV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def comfy_to_cv2(tensor):
    """ComfyUI IMAGE [B,H,W,C] float32 0-1 RGB → numpy uint8 BGR."""
    img = tensor.squeeze(0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cv2_to_comfy(img_bgr):
    """OpenCV BGR uint8 → ComfyUI IMAGE [1,H,W,C] float32 0-1 RGB."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 1: Anime Upscale 4K
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeUpscale4K:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (list(MODELS_REGISTRY.keys()), {"default": "realesr-animevideov3"}),
                "target_resolution": (["4K (3840x2160)", "2K (2560x1440)", "1080p (1920x1080)", "Native x4"],
                                      {"default": "4K (3840x2160)"}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 64}),
                "fp16": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "Upscale anime video frames vers 4K avec Real-ESRGAN. Auto-download du modèle."

    def upscale(self, images, model_name, target_resolution, tile_size, fp16):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        upsampler = load_realesrgan(model_name, tile_size, fp16, device)

        targets = {
            "4K (3840x2160)": (3840, 2160),
            "2K (2560x1440)": (2560, 1440),
            "1080p (1920x1080)": (1920, 1080),
            "Native x4": None,
        }
        target = targets[target_resolution]
        batch_size = images.shape[0]
        results = []

        print(f"[AnimeUpscale] 🔄 Upscaling {batch_size} frames avec {model_name}...")

        for i in range(batch_size):
            frame_bgr = comfy_to_cv2(images[i:i+1])
            try:
                output, _ = upsampler.enhance(frame_bgr, outscale=4)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"[AnimeUpscale] ⚠️  OOM frame {i}, retry tile réduit...")
                    upsampler.tile = max(64, tile_size // 2)
                    output, _ = upsampler.enhance(frame_bgr, outscale=4)
                else:
                    raise

            if target is not None:
                h, w = output.shape[:2]
                src_aspect = w / h
                tgt_aspect = target[0] / target[1]
                if src_aspect >= tgt_aspect:
                    final_w = target[0]
                    final_h = int(round(target[0] / src_aspect / 2) * 2)
                else:
                    final_h = target[1]
                    final_w = int(round(target[1] * src_aspect / 2) * 2)
                if (w, h) != (final_w, final_h):
                    interp = cv2.INTER_AREA if w > final_w else cv2.INTER_LANCZOS4
                    output = cv2.resize(output, (final_w, final_h), interpolation=interp)

            results.append(cv2_to_comfy(output))
            if (i + 1) % 10 == 0 or i == batch_size - 1:
                print(f"[AnimeUpscale]    {i + 1}/{batch_size} frames")

        max_h = max(r.shape[1] for r in results)
        max_w = max(r.shape[2] for r in results)
        padded = []
        for r in results:
            if r.shape[1] != max_h or r.shape[2] != max_w:
                p = torch.zeros(1, max_h, max_w, 3)
                p[:, :r.shape[1], :r.shape[2], :] = r
                padded.append(p)
            else:
                padded.append(r)

        output_batch = torch.cat(padded, dim=0)
        print(f"[AnimeUpscale] ✅ Terminé: {output_batch.shape[2]}x{output_batch.shape[1]}")
        del upsampler
        torch.cuda.empty_cache()
        return (output_batch,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 2: Anime Color Correction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeColorCorrect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05}),
                "color_temperature": ("FLOAT", {"default": 0.0, "min": -0.3, "max": 0.3, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_images",)
    FUNCTION = "correct"
    CATEGORY = CATEGORY
    DESCRIPTION = "Correction couleur pour vidéos anime Wan2.x."

    def correct(self, images, brightness, contrast, saturation, gamma, color_temperature):
        result = images.clone()
        if brightness != 0.0:
            result = result + brightness
        if contrast != 1.0:
            result = (result - 0.5) * contrast + 0.5
        if gamma != 1.0:
            result = torch.clamp(result, 0.0, 1.0)
            result = torch.pow(result, 1.0 / gamma)
        if saturation != 1.0:
            gray = result[..., 0:1] * 0.2989 + result[..., 1:2] * 0.5870 + result[..., 2:3] * 0.1140
            result = gray + (result - gray) * saturation
        if color_temperature != 0.0:
            result[..., 0] = result[..., 0] + color_temperature * 0.5
            result[..., 2] = result[..., 2] - color_temperature * 0.5
        result = torch.clamp(result, 0.0, 1.0)
        return (result,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 3: Anime Sharpen
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeSharpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "radius": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.5}),
                "edge_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sharpened_images",)
    FUNCTION = "sharpen"
    CATEGORY = CATEGORY
    DESCRIPTION = "Sharpening optimisé anime: renforce les lignes sans amplifier le bruit."

    def sharpen(self, images, strength, radius, edge_only):
        if strength == 0:
            return (images,)
        batch_size = images.shape[0]
        results = []
        for i in range(batch_size):
            img_bgr = comfy_to_cv2(images[i:i+1])
            ksize = int(radius * 6) | 1
            blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), radius)
            if edge_only:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_mask = cv2.dilate(edges, None, iterations=2)
                edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 1.0)
                mask = edge_mask.astype(np.float32) / 255.0
                mask = np.stack([mask] * 3, axis=-1)
                sharpened = cv2.addWeighted(img_bgr, 1.0 + strength, blurred, -strength, 0)
                output = (img_bgr.astype(np.float32) * (1 - mask) + sharpened.astype(np.float32) * mask)
                output = output.clip(0, 255).astype(np.uint8)
            else:
                output = cv2.addWeighted(img_bgr, 1.0 + strength, blurred, -strength, 0)
            results.append(cv2_to_comfy(output))
        return (torch.cat(results, dim=0),)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 4: Anime Temporal Denoise
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeTemporalDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "blend_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.8, "step": 0.05}),
                "motion_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_images",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "Réduit le flickering des vidéos Wan2.x par blend temporel adaptatif."

    def denoise(self, images, blend_strength, motion_threshold):
        if blend_strength == 0 or images.shape[0] < 2:
            return (images,)
        result = images.clone()
        batch_size = images.shape[0]
        print(f"[AnimeUpscale] 🔇 Temporal denoise sur {batch_size} frames...")
        for i in range(1, batch_size):
            diff = torch.abs(images[i] - images[i - 1])
            motion = diff.mean(dim=-1, keepdim=True)
            blend_mask = torch.clamp(1.0 - motion / motion_threshold, 0.0, 1.0) * blend_strength
            result[i] = images[i] * (1.0 - blend_mask) + result[i - 1] * blend_mask
        print(f"[AnimeUpscale] ✅ Temporal denoise terminé")
        return (result,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 5: Anime Line Enhancement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeLineEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "line_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.5, "step": 0.05}),
                "line_thickness": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "preserve_color": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_images",)
    FUNCTION = "enhance"
    CATEGORY = CATEGORY
    DESCRIPTION = "Renforce les lignes anime sans toucher aux aplats de couleur."

    def enhance(self, images, line_strength, line_thickness, preserve_color):
        if line_strength == 0:
            return (images,)
        batch_size = images.shape[0]
        results = []
        for i in range(batch_size):
            img_bgr = comfy_to_cv2(images[i:i+1])
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            _, lines = cv2.threshold(gradient, 15, 255, cv2.THRESH_BINARY)
            if line_thickness > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (line_thickness * 2 + 1, line_thickness * 2 + 1))
                lines = cv2.dilate(lines, k, iterations=1)
            lines_smooth = cv2.GaussianBlur(lines, (3, 3), 0.5)
            mask = lines_smooth.astype(np.float32) / 255.0
            if preserve_color:
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                lab[:, :, 0] = lab[:, :, 0] - mask * line_strength * 100
                lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
                output = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            else:
                mask_3ch = np.stack([mask] * 3, axis=-1)
                output = img_bgr.astype(np.float32) * (1.0 - mask_3ch * line_strength)
                output = output.clip(0, 255).astype(np.uint8)
            results.append(cv2_to_comfy(output))
        return (torch.cat(results, dim=0),)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 6: Export Video (FFmpeg)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeExportVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "codec": (["H.265 (HEVC)", "H.264 (AVC)", "AV1 (SVT)"], {"default": "H.265 (HEVC)"}),
                "quality_crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
                "filename_prefix": ("STRING", {"default": "anime_4k"}),
            },
            "optional": {
                "audio_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "export"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Exporte les frames en vidéo MP4 via FFmpeg."

    def export(self, images, fps, codec, quality_crf, filename_prefix, audio_path=""):
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        counter = 1
        while True:
            output_path = os.path.join(output_dir, f"{filename_prefix}_{counter:04d}.mp4")
            if not os.path.exists(output_path):
                break
            counter += 1

        batch_size = images.shape[0]
        h, w = images.shape[1], images.shape[2]
        print(f"[AnimeUpscale] 🎬 Export: {batch_size} frames, {w}x{h}, {fps}fps")

        tmpdir = tempfile.mkdtemp(prefix="anime_export_")
        try:
            for i in range(batch_size):
                frame_bgr = comfy_to_cv2(images[i:i+1])
                cv2.imwrite(os.path.join(tmpdir, f"frame_{i:08d}.png"), frame_bgr)

            ffmpeg_input = os.path.join(tmpdir, "frame_%08d.png")
            codec_map = {
                "H.265 (HEVC)": ["-c:v", "libx265", "-preset", "slow", "-pix_fmt", "yuv420p10le",
                                  "-x265-params", "profile=main10", "-tag:v", "hvc1"],
                "H.264 (AVC)": ["-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p"],
                "AV1 (SVT)": ["-c:v", "libsvtav1", "-preset", "4", "-pix_fmt", "yuv420p10le"],
            }
            cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", ffmpeg_input]
            if audio_path and os.path.isfile(audio_path):
                cmd += ["-i", audio_path, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
            cmd += codec_map[codec]
            cmd += ["-crf", str(quality_crf), "-movflags", "+faststart", output_path]

            print(f"[AnimeUpscale]    Encodage {codec}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr[:300]}")

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[AnimeUpscale] ✅ Vidéo: {output_path} ({size_mb:.1f} Mo)")
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

        return (output_path,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 7: Wan2 Post-Process Pipeline (All-in-One)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Wan2PostProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "upscale_model": (list(MODELS_REGISTRY.keys()), {"default": "realesr-animevideov3"}),
                "target_resolution": (["4K (3840x2160)", "2K (2560x1440)", "1080p (1920x1080)", "Native x4"],
                                      {"default": "4K (3840x2160)"}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 64}),
                "fp16": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "temporal_denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.05}),
                "sharpen_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "line_enhance": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.5, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_images",)
    FUNCTION = "process"
    CATEGORY = CATEGORY
    DESCRIPTION = "Pipeline complète Wan2.x: denoise → color → upscale 4K → lines → sharpen."

    def process(self, images, upscale_model, target_resolution, tile_size, fp16,
                temporal_denoise=0.25, saturation=1.05, contrast=1.05,
                sharpen_strength=0.3, line_enhance=0.3):

        print(f"[AnimeUpscale] 🎌 Pipeline Wan2 ({images.shape[0]} frames)")
        result = images

        if temporal_denoise > 0 and result.shape[0] > 1:
            print(f"[AnimeUpscale]  ① Temporal Denoise ({temporal_denoise})")
            result = AnimeTemporalDenoise().denoise(result, temporal_denoise, 0.1)[0]

        if saturation != 1.0 or contrast != 1.0:
            print(f"[AnimeUpscale]  ② Color Correction")
            result = AnimeColorCorrect().correct(result, 0.0, contrast, saturation, 1.0, 0.0)[0]

        print(f"[AnimeUpscale]  ③ Upscale → {target_resolution}")
        result = AnimeUpscale4K().upscale(result, upscale_model, target_resolution, tile_size, fp16)[0]

        if line_enhance > 0:
            print(f"[AnimeUpscale]  ④ Line Enhancement ({line_enhance})")
            result = AnimeLineEnhance().enhance(result, line_enhance, 1, True)[0]

        if sharpen_strength > 0:
            print(f"[AnimeUpscale]  ⑤ Sharpen ({sharpen_strength})")
            result = AnimeSharpen().sharpen(result, sharpen_strength, 1.0, True)[0]

        print(f"[AnimeUpscale] 🎉 Pipeline terminée: {result.shape[2]}x{result.shape[1]}")
        return (result,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ComfyUI Registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE_CLASS_MAPPINGS = {
    "AnimeUpscale4K": AnimeUpscale4K,
    "AnimeColorCorrect": AnimeColorCorrect,
    "AnimeSharpen": AnimeSharpen,
    "AnimeTemporalDenoise": AnimeTemporalDenoise,
    "AnimeLineEnhance": AnimeLineEnhance,
    "AnimeExportVideo": AnimeExportVideo,
    "Wan2PostProcess": Wan2PostProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimeUpscale4K": "🎌 Anime Upscale 4K",
    "AnimeColorCorrect": "🎨 Anime Color Correct",
    "AnimeSharpen": "✨ Anime Sharpen",
    "AnimeTemporalDenoise": "🔇 Anime Temporal Denoise",
    "AnimeLineEnhance": "✏️ Anime Line Enhance",
    "AnimeExportVideo": "🎬 Anime Export Video",
    "Wan2PostProcess": "⚡ Wan2 Post-Process Pipeline",
}
