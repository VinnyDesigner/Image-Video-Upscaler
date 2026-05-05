import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from utils import download_model
from config import MODELS_DIR, DEFAULT_MODEL

class AIUpscaler:
    def __init__(self, model_name=DEFAULT_MODEL, scale=4, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
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
            half=USE_HALF if self.device.type == 'cuda' else False,
            device=self.device
        )

    def upscale(self, img_input):
        if isinstance(img_input, str):
            img = cv2.imread(img_input, cv2.IMREAD_UNCHANGED)
        else:
            img = img_input
            
        output, _ = self.upsampler.enhance(img, outscale=self.scale)
        return output
