import os
import torch
import cv2
from frame_extractor import FrameExtractor
from upscaler import AIUpscaler
from face_enhancer import FaceEnhancer
from video_writer import VideoWriter
from utils import get_video_metadata, logger
from config import TEMP_DIR, OUTPUTS_DIR
import shutil
from concurrent.futures import ThreadPoolExecutor
import time

class UpscalePipeline:
    def __init__(self, video_path, scale=4, face_enhance=True, bitrate="8M", rotation=0):
        """Initialize the upscaling pipeline.

        Ensures that a CUDA‑compatible GPU is available; otherwise raises an error.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA‑compatible GPU not detected. Please ensure GPU drivers and a CUDA‑enabled PyTorch are installed.")
        self.video_path = video_path
        self.scale = scale
        self.face_enhance = face_enhance
        self.bitrate = bitrate
        self.rotation = rotation
        
        self.metadata = get_video_metadata(video_path)
        if self.metadata is None:
            raise ValueError("Failed to read video metadata. The file may be corrupt or not a valid video format.")
            
        self.extractor = FrameExtractor(video_path, manual_rotation=rotation)
        self.writer = VideoWriter(video_path)
        
        # Optimized Model Selection:
        # If face enhancement is on, GFPGAN handles both upscaling and faces.
        # Otherwise, Real-ESRGAN handles the upscaling.
        if face_enhance:
            self.processor = FaceEnhancer(upscale=scale)
        else:
            self.processor = AIUpscaler(scale=scale)

    def run(self, progress_callback=None):
        logger.info(f"Starting pipeline for {self.video_path}")
        
        # 1. Extract Frames
        total_frames = self.metadata['frame_count']
        frames_dir = self.extractor.temp_folder
        upscaled_dir = frames_dir / "upscaled"
        upscaled_dir.mkdir(exist_ok=True)
        
        count = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, frame_path in self.extractor.extract_frames():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Read frame once
                img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                
                # Process Frame (single optimized pass)
                enhanced_frame = self.processor.enhance(img) if hasattr(self.processor, 'enhance') else self.processor.upscale(img)
                
                # Save upscaled frame in background
                out_path = upscaled_dir / os.path.basename(frame_path)
                executor.submit(cv2.imwrite, str(out_path), enhanced_frame)
                
                count += 1
                if progress_callback and count % 5 == 0: # Update UI every 5 frames to reduce overhead
                    progress_callback(count / total_frames, f"Processing frame {count}/{total_frames} ({ (count/total_frames)*100:.1f}%)")

        # 3. Assemble Video
        logger.info("Assembling video...")
        pattern = str(upscaled_dir / "frame_%06d.png")
        output_path = self.writer.assemble_video(pattern, self.metadata['fps'], bitrate=self.bitrate)
        
        # 4. Cleanup
        self.extractor.cleanup()
        
        return output_path
