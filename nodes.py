"""
ComfyUI-AnimeUpscale4K
━━━━━━━━━━━━━━━━━━━━━━
Custom node pack for post-processing anime-style videos generated with Wan2.1/2.2.
Includes: Real-ESRGAN upscaling (4K), color correction, sharpening, temporal denoising,
           and video export with FFmpeg.

All models are auto-downloaded on first use.
"""

import hashlib
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import folder_paths  # ComfyUI utility

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
        "description": "Optimisé vidéo anime (rapide, bon temporel)",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "num_block": 6,
        "num_grow_ch": 32,
        "description": "Haute qualité images anime",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utility: Model downloader
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
                print(f"\r[AnimeUpscale]    {pct:.1f}% ({downloaded // 1024}KB / {total // 1024}KB)", end="", flush=True)
    print()

    os.rename(tmp_path, model_path)
    print(f"[AnimeUpscale] ✅ Modèle sauvegardé: {model_path}")
    return model_path


def load_realesrgan(model_name: str, tile: int, fp16: bool, device: str):
    """Load Real-ESRGAN upsampler."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model_path = ensure_model(model_name)
    info = MODELS_REGISTRY[model_name]

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=info["num_block"],
        num_grow_ch=info["num_grow_ch"],
        scale=4,
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=fp16 and device == "cuda",
        device=device,
    )
    return upsampler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers: ComfyUI tensor ↔ OpenCV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def comfy_to_cv2(tensor):
    """ComfyUI IMAGE tensor [B,H,W,C] float32 0-1 RGB → numpy uint8 BGR."""
    img = tensor.squeeze(0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cv2_to_comfy(img_bgr):
    """OpenCV BGR uint8 → ComfyUI IMAGE tensor [1,H,W,C] float32 0-1 RGB."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 1: Anime Upscale 4K (Real-ESRGAN)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeUpscale4K:
    """
    Upscale anime frames vers 4K avec Real-ESRGAN.
    Accepte le batch IMAGE de ComfyUI (ex: frames Wan2.1/2.2).
    Auto-télécharge le modèle au premier usage.
    """

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

        # Parse target
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
                    print(f"[AnimeUpscale] ⚠️  OOM frame {i}, retry avec tile réduit...")
                    upsampler.tile = max(64, tile_size // 2)
                    output, _ = upsampler.enhance(frame_bgr, outscale=4)
                else:
                    raise

            # Resize vers la cible si nécessaire
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
                    # Downscale = supersampling (meilleur), upscale = Lanczos
                    interp = cv2.INTER_AREA if w > final_w else cv2.INTER_LANCZOS4
                    output = cv2.resize(output, (final_w, final_h), interpolation=interp)

            results.append(cv2_to_comfy(output))

            if (i + 1) % 10 == 0 or i == batch_size - 1:
                print(f"[AnimeUpscale]    {i + 1}/{batch_size} frames traitées")

        # Toutes les frames doivent avoir la même taille pour le batch
        # Pad si nécessaire (cas de résolutions source variables)
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
        print(f"[AnimeUpscale] ✅ Upscaling terminé: {output_batch.shape[1]}x{output_batch.shape[2]}")

        # Cleanup
        del upsampler
        torch.cuda.empty_cache()

        return (output_batch,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 2: Anime Color Correction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeColorCorrect:
    """
    Correction colorimétrique optimisée pour les vidéos anime.
    Corrige les artefacts de couleur courants des modèles Wan2.x.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05}),
                "color_temperature": ("FLOAT", {"default": 0.0, "min": -0.3, "max": 0.3, "step": 0.01,
                                                  "tooltip": "Négatif=froid/bleu, Positif=chaud/orange"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_images",)
    FUNCTION = "correct"
    CATEGORY = CATEGORY
    DESCRIPTION = "Correction couleur pour vidéos anime Wan2.x: luminosité, contraste, saturation, gamma, température."

    def correct(self, images, brightness, contrast, saturation, gamma, color_temperature):
        result = images.clone()

        # Brightness
        if brightness != 0.0:
            result = result + brightness

        # Contrast (around 0.5 midpoint)
        if contrast != 1.0:
            result = (result - 0.5) * contrast + 0.5

        # Gamma
        if gamma != 1.0:
            result = torch.clamp(result, 0.0, 1.0)
            result = torch.pow(result, 1.0 / gamma)

        # Saturation (in-place for memory)
        if saturation != 1.0:
            gray = result[..., 0:1] * 0.2989 + result[..., 1:2] * 0.5870 + result[..., 2:3] * 0.1140
            result = gray + (result - gray) * saturation

        # Color temperature shift
        if color_temperature != 0.0:
            result[..., 0] = result[..., 0] + color_temperature * 0.5   # R
            result[..., 2] = result[..., 2] - color_temperature * 0.5   # B

        result = torch.clamp(result, 0.0, 1.0)
        return (result,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 3: Anime Sharpen
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeSharpen:
    """
    Sharpening adapté aux lignes nettes de l'anime.
    Utilise Unsharp Mask pour renforcer les contours sans amplifier le bruit.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                                        "tooltip": "0=pas de sharpen, 0.5=subtil, 1.0=fort"}),
                "radius": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.5,
                                      "tooltip": "Rayon du flou gaussien (sigma)"}),
                "edge_only": ("BOOLEAN", {"default": False,
                                           "tooltip": "Sharpen uniquement les contours (préserve les aplats)"}),
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

            # Gaussian blur for unsharp mask
            ksize = int(radius * 6) | 1  # Ensure odd
            blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), radius)

            if edge_only:
                # Detect edges, only sharpen near them
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
    """
    Débruitage temporel pour réduire le scintillement (flickering)
    courant dans les vidéos générées par Wan2.x.
    Moyenne pondérée entre frames consécutives.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "blend_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.8, "step": 0.05,
                                              "tooltip": "0=pas de blend, 0.3=subtil, 0.6=fort (risque ghosting)"}),
                "motion_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01,
                                                "tooltip": "Seuil de mouvement: au-dessus, pas de blend (évite ghosting)"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_images",)
    FUNCTION = "denoise"
    CATEGORY = CATEGORY
    DESCRIPTION = "Réduit le scintillement (flickering) des vidéos Wan2.x par blend temporel adaptatif."

    def denoise(self, images, blend_strength, motion_threshold):
        if blend_strength == 0 or images.shape[0] < 2:
            return (images,)

        result = images.clone()
        batch_size = images.shape[0]

        print(f"[AnimeUpscale] 🔇 Temporal denoise sur {batch_size} frames (strength={blend_strength})...")

        for i in range(1, batch_size):
            # Compute per-pixel difference
            diff = torch.abs(images[i] - images[i - 1])
            motion = diff.mean(dim=-1, keepdim=True)  # [H, W, 1]

            # Adaptive blend: less blending where motion is high
            blend_mask = torch.clamp(1.0 - motion / motion_threshold, 0.0, 1.0) * blend_strength

            # Weighted average with previous frame
            result[i] = images[i] * (1.0 - blend_mask) + result[i - 1] * blend_mask

        print(f"[AnimeUpscale] ✅ Temporal denoise terminé")
        return (result,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 5: Anime Line Enhancement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeLineEnhance:
    """
    Renforce les lignes de dessin anime (line art) sans affecter
    les aplats de couleur. Idéal après upscaling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "line_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.5, "step": 0.05}),
                "line_thickness": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "preserve_color": ("BOOLEAN", {"default": True,
                                                "tooltip": "Préserver les couleurs originales (ne modifie que la luminance)"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_images",)
    FUNCTION = "enhance"
    CATEGORY = CATEGORY
    DESCRIPTION = "Renforce les lignes de dessin anime sans toucher aux aplats de couleur."

    def enhance(self, images, line_strength, line_thickness, preserve_color):
        if line_strength == 0:
            return (images,)

        batch_size = images.shape[0]
        results = []

        for i in range(batch_size):
            img_bgr = comfy_to_cv2(images[i:i+1])

            # Extract edges
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Use morphological gradient for clean anime lines
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

            # Threshold to get clean lines
            _, lines = cv2.threshold(gradient, 15, 255, cv2.THRESH_BINARY)

            # Thicken lines
            if line_thickness > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (line_thickness * 2 + 1, line_thickness * 2 + 1))
                lines = cv2.dilate(lines, k, iterations=1)

            # Smooth the mask
            lines_smooth = cv2.GaussianBlur(lines, (3, 3), 0.5)
            mask = lines_smooth.astype(np.float32) / 255.0

            if preserve_color:
                # Darken lines in luminance channel only (LAB)
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                lab[:, :, 0] = lab[:, :, 0] - mask * line_strength * 100
                lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
                output = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            else:
                # Simple darkening
                mask_3ch = np.stack([mask] * 3, axis=-1)
                output = img_bgr.astype(np.float32) * (1.0 - mask_3ch * line_strength)
                output = output.clip(0, 255).astype(np.uint8)

            results.append(cv2_to_comfy(output))

        return (torch.cat(results, dim=0),)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 6: Export Video (FFmpeg)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnimeExportVideo:
    """
    Exporte un batch d'images ComfyUI en vidéo MP4/MKV via FFmpeg.
    Supporte H.265, H.264, AV1 avec paramètres de qualité.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "codec": (["H.265 (HEVC)", "H.264 (AVC)", "AV1 (SVT)"], {"default": "H.265 (HEVC)"}),
                "quality_crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1,
                                         "tooltip": "0=lossless, 18=haute qualité, 28=qualité web"}),
                "filename_prefix": ("STRING", {"default": "anime_4k"}),
            },
            "optional": {
                "audio_path": ("STRING", {"default": "", "tooltip": "Chemin vers fichier audio à intégrer"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "export"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Exporte les frames en vidéo MP4 via FFmpeg (H.265/H.264/AV1)."

    def export(self, images, fps, codec, quality_crf, filename_prefix, audio_path=""):
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # Find unique filename
        counter = 1
        while True:
            output_path = os.path.join(output_dir, f"{filename_prefix}_{counter:04d}.mp4")
            if not os.path.exists(output_path):
                break
            counter += 1

        batch_size = images.shape[0]
        h, w = images.shape[1], images.shape[2]

        print(f"[AnimeUpscale] 🎬 Export vidéo: {batch_size} frames, {w}x{h}, {fps}fps")

        # Write frames to temp directory
        tmpdir = tempfile.mkdtemp(prefix="anime_export_")
        try:
            for i in range(batch_size):
                frame_bgr = comfy_to_cv2(images[i:i+1])
                frame_path = os.path.join(tmpdir, f"frame_{i:08d}.png")
                cv2.imwrite(frame_path, frame_bgr)

            # Build FFmpeg command
            ffmpeg_input = os.path.join(tmpdir, "frame_%08d.png")

            codec_map = {
                "H.265 (HEVC)": ["-c:v", "libx265", "-preset", "slow", "-pix_fmt", "yuv420p10le",
                                  "-x265-params", "profile=main10", "-tag:v", "hvc1"],
                "H.264 (AVC)": ["-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p"],
                "AV1 (SVT)": ["-c:v", "libsvtav1", "-preset", "4", "-pix_fmt", "yuv420p10le"],
            }

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", ffmpeg_input,
            ]

            # Audio
            if audio_path and os.path.isfile(audio_path):
                cmd += ["-i", audio_path, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]

            cmd += codec_map[codec]
            cmd += ["-crf", str(quality_crf)]
            cmd += ["-movflags", "+faststart", output_path]

            print(f"[AnimeUpscale]    Encodage {codec}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode != 0:
                print(f"[AnimeUpscale] ❌ FFmpeg error:\n{result.stderr[:500]}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr[:200]}")

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[AnimeUpscale] ✅ Vidéo exportée: {output_path} ({size_mb:.1f} Mo)")

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

        return (output_path,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 7: Wan2 Post-Process Pipeline (All-in-One)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Wan2PostProcess:
    """
    Pipeline tout-en-un de post-traitement pour Wan2.1/2.2:
    Temporal Denoise → Color Correct → Upscale 4K → Line Enhance → Sharpen

    Combine tous les nœuds en un seul pour un workflow simplifié.
    """

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
    DESCRIPTION = "Pipeline complète Wan2.x: denoise → color → upscale 4K → lines → sharpen. Tout-en-un."

    def process(self, images, upscale_model, target_resolution, tile_size, fp16,
                temporal_denoise=0.25, saturation=1.05, contrast=1.05,
                sharpen_strength=0.3, line_enhance=0.3):

        print(f"[AnimeUpscale] 🎌 Pipeline Wan2 Post-Process ({images.shape[0]} frames)")
        result = images

        # Step 1: Temporal denoise
        if temporal_denoise > 0 and result.shape[0] > 1:
            print(f"[AnimeUpscale]  ① Temporal Denoise (strength={temporal_denoise})")
            node = AnimeTemporalDenoise()
            result = node.denoise(result, temporal_denoise, 0.1)[0]

        # Step 2: Color correction
        if saturation != 1.0 or contrast != 1.0:
            print(f"[AnimeUpscale]  ② Color Correction (sat={saturation}, contrast={contrast})")
            node = AnimeColorCorrect()
            result = node.correct(result, 0.0, contrast, saturation, 1.0, 0.0)[0]

        # Step 3: Upscale
        print(f"[AnimeUpscale]  ③ Upscale → {target_resolution}")
        node = AnimeUpscale4K()
        result = node.upscale(result, upscale_model, target_resolution, tile_size, fp16)[0]

        # Step 4: Line enhancement
        if line_enhance > 0:
            print(f"[AnimeUpscale]  ④ Line Enhancement (strength={line_enhance})")
            node = AnimeLineEnhance()
            result = node.enhance(result, line_enhance, 1, True)[0]

        # Step 5: Sharpen
        if sharpen_strength > 0:
            print(f"[AnimeUpscale]  ⑤ Sharpen (strength={sharpen_strength})")
            node = AnimeSharpen()
            result = node.sharpen(result, sharpen_strength, 1.0, True)[0]

        print(f"[AnimeUpscale] 🎉 Pipeline terminée: {result.shape[1]}x{result.shape[2]}")
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
