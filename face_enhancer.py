import cv2
import torch
from gfpgan import GFPGANer
from utils import download_model
from config import MODELS_DIR, FACE_ENHANCE_MODEL

class FaceEnhancer:
    def __init__(self, model_name=FACE_ENHANCE_MODEL, upscale=2, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model_path = download_model(model_name)
        
        self.restorer = GFPGANer(
            model_path=self.model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            device=self.device
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
