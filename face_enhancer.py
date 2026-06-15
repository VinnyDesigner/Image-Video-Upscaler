import cv2
import torch
from gfpgan import GFPGANer
from utils import download_model
from config import MODELS_DIR, FACE_ENHANCE_MODEL

class FaceEnhancer:
    def __init__(self, model_name=FACE_ENHANCE_MODEL, upscale=2, device=None, prefer_gpu: bool = True):
        """Initialize the face enhancer.

        Args:
            model_name (str): Model identifier.
            upscale (int): Upscaling factor.
            device (str|torch.device, optional): Explicit device; auto‑detect if None.
            prefer_gpu (bool): Enforce GPU usage; raise RuntimeError if CUDA not available.
        """
        if device is None:
            if prefer_gpu and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA‑compatible GPU not detected. Install NVIDIA drivers and a CUDA‑enabled PyTorch build, or set `prefer_gpu=False` to fallback to CPU."
                )
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

        self.model_path = download_model(model_name)
        
        self.restorer = GFPGANer(
            model_path=self.model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            device=self._device
        )

    def enhance(self, img):
        # img should be a numpy array (BGR)
        _, _, restored_img = self.restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        return restored_img
