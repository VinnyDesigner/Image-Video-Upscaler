import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from utils import download_model
from config import MODELS_DIR, DEFAULT_MODEL

class AIUpscaler:
    def __init__(self, model_name=DEFAULT_MODEL, scale=4, device=None, prefer_gpu: bool = True):
        """Initialize the upscaler.

        Args:
            model_name (str): Name of the model to download.
            scale (int): Upscaling factor.
            device (str|torch.device, optional): Explicit device string; if None, auto‑detect.
            prefer_gpu (bool): If True, enforce GPU usage and raise an error when CUDA is unavailable.
        """
        if device is None:
            # Auto‑detect device but respect `prefer_gpu`
            if prefer_gpu and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA‑compatible GPU not detected. Install appropriate NVIDIA drivers and a CUDA‑enabled PyTorch build, or set `prefer_gpu=False` to fallback to CPU."
                )
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

            
        self.model_path = download_model(model_name)
        self.scale = scale
        
        # Initialize model architecture
        # RealESRGAN_x4plus uses RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        from config import TILE_SIZE, USE_HALF
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_path,
            model=model,
            tile=TILE_SIZE,
            tile_pad=10,
            pre_pad=0,
            half=USE_HALF if self._device.type == 'cuda' else False,
            device=self._device
        )

    @property
    def device(self):
        return self._device

    def upscale(self, img_input):
        if isinstance(img_input, str):
            img = cv2.imread(img_input, cv2.IMREAD_UNCHANGED)
        else:
            img = img_input
            
        output, _ = self.upsampler.enhance(img, outscale=self.scale)
        return output
