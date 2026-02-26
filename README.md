# 🎌 ComfyUI-AnimeUpscale4K

**Post-processing node pack for anime-style videos generated with Wan2.1/2.2.**

Upscale to 4K with Real-ESRGAN, fix flickering, enhance colors and line art — all with auto model download.

---

## 📦 Nodes

| Node | Description |
|------|-------------|
| **🎌 Anime Upscale 4K** | Real-ESRGAN upscaling (auto model download) → 4K/2K/1080p |
| **🎨 Anime Color Correct** | Brightness, contrast, saturation, gamma, color temperature |
| **✨ Anime Sharpen** | Unsharp mask optimized for anime (edge-only mode) |
| **🔇 Anime Temporal Denoise** | Reduce Wan2.x flickering via adaptive temporal blend |
| **✏️ Anime Line Enhance** | Reinforce line art in luminance (preserves flat colors) |
| **🎬 Anime Export Video** | Export to MP4 via FFmpeg (H.265/H.264/AV1 + audio) |
| **⚡ Wan2 Post-Process Pipeline** | All-in-one: denoise → color → upscale → lines → sharpen |

---

## 🚀 Installation

### Via ComfyUI Manager (recommandé)
Search for `AnimeUpscale4K` in ComfyUI Manager and click Install.

### Manuel
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-AnimeUpscale4K.git
cd ComfyUI-AnimeUpscale4K
pip install -r requirements.txt
```

### Prérequis
- **Python** 3.10+
- **PyTorch** 2.0+ avec CUDA
- **FFmpeg** installé (pour le nœud Export Video)
- **GPU** 6+ Go VRAM recommandé (réduire `tile_size` si nécessaire)

---

## 🎯 Workflow recommandé pour Wan2.2

### Simple (All-in-One)
```
[Wan2.2 Generate] → [⚡ Wan2 Post-Process Pipeline] → [🎬 Export Video]
```

### Avancé (contrôle total)
```
[Wan2.2 Generate]
    ↓
[🔇 Temporal Denoise]    ← Réduit le flickering
    ↓
[🎨 Color Correct]       ← Ajuste saturation, contraste
    ↓
[🎌 Anime Upscale 4K]    ← Upscale Real-ESRGAN → 4K
    ↓
[✏️ Line Enhance]        ← Renforce les lignes
    ↓
[✨ Anime Sharpen]        ← Sharpen final
    ↓
[🎬 Export Video]         ← MP4 H.265
```

---

## ⚙️ Paramètres recommandés par résolution source

| Source Wan2.2 | Upscale Target | Tile Size | Qualité |
|---------------|----------------|-----------|---------|
| 480×320 | 4K | 256 | Bon |
| 720×480 | 4K | 256 | Très bon |
| 1280×720 | 4K | 256-512 | Excellent |
| 1920×1080 | 2K ou 4K | 512 | Parfait |

---

## 💡 Tips

- **VRAM insuffisante ?** Réduisez `tile_size` à 128 ou 64.
- **Flickering ?** Augmentez `temporal_denoise` (0.3-0.5), mais attention au ghosting.
- **Couleurs ternes ?** Montez `saturation` à 1.1-1.2 et `contrast` à 1.05-1.1.
- **Lignes floues ?** Utilisez `Line Enhance` (0.3-0.5) + `Sharpen edge_only` (0.3-0.5).
- **Export léger ?** Utilisez AV1 avec CRF 24-28 pour une taille réduite.

---

## 📋 Models (auto-downloaded)

Les modèles sont téléchargés automatiquement dans `ComfyUI/models/anime_upscale/` :

- `realesr-animevideov3.pth` (~16 Mo) — Optimisé vidéo anime
- `RealESRGAN_x4plus_anime_6B.pth` (~16 Mo) — Haute qualité image anime

---

## 📄 License

MIT License
