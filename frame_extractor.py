import cv2
import os
import ffmpeg
from pathlib import Path
from config import TEMP_DIR
import shutil

class FrameExtractor:
    def __init__(self, video_path, manual_rotation=0):
        self.video_path = video_path
        self.manual_rotation = manual_rotation
        self.temp_folder = TEMP_DIR / Path(video_path).stem
        if self.temp_folder.exists():
            shutil.rmtree(self.temp_folder)
        self.temp_folder.mkdir(parents=True, exist_ok=True)

    def extract_frames(self):
        import time
        # Build the FFmpeg command
        vf_chain = []
        if self.manual_rotation == 90:
            vf_chain.append("transpose=1")
        elif self.manual_rotation == 180:
            vf_chain.append("transpose=1,transpose=1")
        elif self.manual_rotation == 270:
            vf_chain.append("transpose=2")
        
        input_stream = ffmpeg.input(self.video_path)
        vf_str = ",".join(vf_chain) if vf_chain else None
        
        if vf_str:
            output_stream = input_stream.output(str(self.temp_folder / "frame_%06d.png"), vf=vf_str, start_number=0)
        else:
            output_stream = input_stream.output(str(self.temp_folder / "frame_%06d.png"), start_number=0)

        # Run asynchronously
        process = output_stream.overwrite_output().run_async()
        
        frame_index = 0
        # Wait a bit for the first frame
        timeout = 30 # 30 seconds timeout for first frame
        start_time = time.time()
        
        while True:
            frame_path = self.temp_folder / f"frame_{frame_index:06d}.png"
            if frame_path.exists():
                # Small delay to ensure the file is fully written
                # (Though FFmpeg usually writes quickly, PNGs can take a moment)
                yield frame_index + 1, str(frame_path)
                frame_index += 1
                start_time = time.time() # Reset timeout
            else:
                if process.poll() is not None:
                    # Check if we missed any frames written at the very end
                    time.sleep(0.5)
                    if not frame_path.exists():
                        break
                
                if time.time() - start_time > timeout:
                    logger.error("FFmpeg extraction timed out or failed.")
                    process.kill()
                    break
                    
                time.sleep(0.1)

        process.wait()

    def cleanup(self):
        if self.temp_folder.exists():
            shutil.rmtree(self.temp_folder)
