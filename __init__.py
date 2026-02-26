"""
ComfyUI-AnimeUpscale4K
━━━━━━━━━━━━━━━━━━━━━━
Post-processing nodes for anime video upscaling to 4K.
Optimized for Wan2.1/2.2 generated videos.

Nodes:
  🎌 Anime Upscale 4K        — Real-ESRGAN upscaling with auto model download
  🎨 Anime Color Correct      — Brightness, contrast, saturation, gamma, temperature
  ✨ Anime Sharpen             — Unsharp mask optimized for anime lines
  🔇 Anime Temporal Denoise   — Reduce flickering from Wan2.x
  ✏️ Anime Line Enhance       — Reinforce line art without affecting flat colors
  🎬 Anime Export Video        — Export to MP4 via FFmpeg (H.265/H.264/AV1)
  ⚡ Wan2 Post-Process Pipeline — All-in-one: denoise → color → upscale → lines → sharpen
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
