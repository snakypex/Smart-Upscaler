# 🔍 Smart Upscaler — ComfyUI Custom Node

Upscale des frames issues d'interpolation vers **1080p, 2K, 4K ou 8K** via RealESRGAN sous **CUDA**.  
Gère tous les formats : paysage, portrait, carré, ultra-wide, etc.

---

## ✨ Fonctionnalités

| Feature | Détail |
|---|---|
| **Résolutions cibles** | 1080p · 2K · 4K · 8K (bord long) |
| **Formats supportés** | Horizontal, Vertical, Carré — ratio conservé automatiquement |
| **CUDA** | Half-precision (fp16) pour vitesse maximale |
| **Auto-download** | Le modèle se télécharge automatiquement au premier usage |
| **Tiling** | Traitement par tuiles → pas d'OOM même sur les grandes frames |
| **Multi-pass** | Plusieurs passes si le facteur d'agrandissement requis est > 4× |
| **Batch** | Traite les batches de frames (sorties d'interpolation) |

---

## 📦 Installation

```bash
# 1. Copier dans le dossier custom_nodes de ComfyUI
cp -r comfyui_smart_upscaler/ <ComfyUI>/custom_nodes/

# 2. Installer les dépendances (si pas déjà présentes)
pip install torch torchvision tqdm requests
```

Relancer ComfyUI — le nœud apparaît dans la catégorie **image/upscaling**.

---

## 🤖 Modèles disponibles

| Modèle | Facteur | Usage recommandé | Téléchargement automatique |
|---|---|---|---|
| `RealESRGAN-x4plus` | ×4 | Vidéo réaliste, photos | ✅ |
| `RealESRGAN-x2plus` | ×2 | Upscale modéré, qualité max | ✅ |
| `RealESRGAN-animevideo-x4` | ×4 | Anime, cartoon, illustration | ✅ |

Les fichiers `.pth` sont sauvegardés dans `models/upscale_models/`.

---

## 🔌 Paramètres du nœud

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `image` | IMAGE | — | Entrée : frames interpolées (batch OK) |
| `target_resolution` | Enum | `4K` | Résolution cible (bord long en pixels) |
| `model_name` | Enum | `RealESRGAN-x4plus` | Modèle d'upscaling |
| `tile_size` | INT | `512` | Taille des tuiles CUDA (baisser si OOM) |
| `tile_overlap` | INT | `32` | Chevauchement des tuiles (cache les jointures) |
| `force_exact_resolution` | BOOL | `False` | Force crop carré exact (rare) |

---

## 🔄 Exemple de workflow

```
[Video Loader] → [Frame Interpolation] → [SmartUpscaler 4K] → [Video Combine]
```

Ou en batch :
```
[Image Batch] → [SmartUpscaler 2K] → [Preview / Save]
```

---

## ⚡ Performances indicatives (RTX 3090, fp16)

| Frame source | Cible | Temps/frame |
|---|---|---|
| 540p → 4K | ×8 (2 passes) | ~1.2s |
| 1080p → 4K | ×2 + resize | ~0.4s |
| 720p → 8K | ×12 (3 passes) | ~3.5s |

---

## 🛠 Dépannage

**CUDA OOM** → Réduire `tile_size` (ex: 256)  
**Frames floues** → Augmenter `tile_overlap` (ex: 64)  
**Téléchargement bloqué** → Télécharger manuellement le `.pth` dans `models/upscale_models/`  
**CPU lent** → Installer CUDA + PyTorch GPU : `pip install torch --index-url https://download.pytorch.org/whl/cu121`
