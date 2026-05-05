import os
from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).parent.absolute()

# Directories
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp_frames"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in [MODELS_DIR, TEMP_DIR, OUTPUTS_DIR]:
    d.mkdir(exist_ok=True)

# Model Settings
DEFAULT_MODEL = "RealESRGAN_x4plus"  # Options: RealESRGAN_x4plus, RealESRNet_x4plus, etc.
FACE_ENHANCE_MODEL = "GFPGANv1.4"

# Processing Settings
BATCH_SIZE = 4  # Frames per batch for GPU (not currently used by libraries)
THREADS = 4     # For CPU fallback and disk I/O
TILE_SIZE = 800 # Higher = faster but more VRAM usage (default 400-800)
USE_HALF = True # Use FP16 for speed (CUDA only)

# UI Configuration
THEME = "dark"
APP_TITLE = "AI Video Upscaler Pro"

# Video Quality Settings
PRESET_RESOLUTIONS = {
    "2K": (2560, 1440),
    "4K": (3840, 2160)
}

BITRATE_OPTIONS = {
    "Low": "2M",
    "Medium": "8M",
    "High": "20M"
}
