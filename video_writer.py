import ffmpeg
import os
from pathlib import Path
from config import OUTPUTS_DIR

class VideoWriter:
    def __init__(self, original_video, output_name=None):
        self.original_video = original_video
        if output_name:
            self.output_path = OUTPUTS_DIR / output_name
        else:
            self.output_path = OUTPUTS_DIR / f"{Path(original_video).stem}_upscaled.mp4"

    def assemble_video(self, frame_pattern, fps, bitrate="8M"):
        # Pattern example: "temp_frames/my_video/frame_%06d.png"
        
        input_frames = ffmpeg.input(frame_pattern, framerate=fps)
        
        # Check if original video has audio
        try:
            probe = ffmpeg.probe(self.original_video)
            has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        except:
            has_audio = False

        if has_audio:
            input_audio = ffmpeg.input(self.original_video).audio
            output = ffmpeg.output(input_frames, input_audio, str(self.output_path), 
                                  vcodec='libx264', pix_fmt='yuv420p', video_bitrate=bitrate)
        else:
            output = ffmpeg.output(input_frames, str(self.output_path), 
                                  vcodec='libx264', pix_fmt='yuv420p', video_bitrate=bitrate)
        
        # Combine frames and audio
        try:
            process = (
                output
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return str(self.output_path)
        except ffmpeg.Error as e:
            print('--- FFmpeg Error Output ---')
            print(e.stderr.decode('utf8'))
            raise e
