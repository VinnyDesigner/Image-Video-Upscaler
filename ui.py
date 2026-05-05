import gradio as gr
import os
import cv2
import numpy as np
from pipeline import UpscalePipeline
from upscaler import AIUpscaler
from face_enhancer import FaceEnhancer
from color_correction import apply_color_correction
from outpainting import extend_image
from utils import get_video_metadata, logger
from config import OUTPUTS_DIR
import time
import uuid

def process_video(video_file, scale, face_enhance, bitrate, rotation, progress=gr.Progress()):
    if video_file is None:
        return None, "Please upload a video."
    
    start_time = time.time()
    
    def update_progress(percent, message):
        progress(percent, desc=message)
    
    # Map rotation string to degrees
    rotation_map = {"None": 0, "90° CW": 90, "180°": 180, "90° CCW": 270}
    rot_val = rotation_map.get(rotation, 0)
    
    pipeline = UpscalePipeline(
        video_path=video_file,
        scale=int(scale.replace('x', '')),
        face_enhance=face_enhance,
        bitrate=bitrate,
        rotation=rot_val
    )
    
    output_path = pipeline.run(progress_callback=update_progress)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return output_path, f"Processing complete in {duration:.2f} seconds."

def get_info(video_file):
    if video_file is None:
        return "No video uploaded."
    meta = get_video_metadata(video_file)
    return f"""
    ### Video Metadata
    - **Resolution:** {meta['width']}x{meta['height']}
    - **FPS:** {meta['fps']:.2f}
    - **Duration:** {meta['duration']:.2f}s
    - **Size:** {meta['size_mb']:.2f} MB
    """

def process_image(image, scale, face_enhance, enable_color, color_mode, enable_outpaint, out_w, out_h, out_pad, out_mode):
    if image is None:
        return None, "Please upload an image."
    
    start_time = time.time()
    
    # 1. BGR Conversion
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 2. Optional Color Correction (Pre-Upscale)
    if enable_color:
        img_bgr = apply_color_correction(img_bgr, mode=color_mode)
    
    # 3. Optimized Processing (Single Pass)
    scale_factor = int(scale.replace('x', ''))
    
    if face_enhance:
        processor = FaceEnhancer(upscale=scale_factor)
        enhanced = processor.enhance(img_bgr)
    else:
        processor = AIUpscaler(scale=scale_factor)
        enhanced = processor.upscale(img_bgr)
    
    # 4. Optional Image Extension (Post-Upscale)
    if enable_outpaint:
        enhanced = extend_image(enhanced, target_width=out_w, target_height=out_h, padding=int(out_pad), mode=out_mode)
    
    # Save output
    output_filename = f"enhanced_{uuid.uuid4()}.png"
    output_path = OUTPUTS_DIR / output_filename
    cv2.imwrite(str(output_path), enhanced)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), f"Enhanced in {duration:.2f} seconds."

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), title="AI Video Upscaler Pro") as demo:
    gr.Markdown("""
    # ✨ AI Media Upscaler Pro
    ### Professional-Grade AI Enhancement for Video & Images
    """)
    
    with gr.Tabs():
        # --- Video Tab ---
        with gr.Tab("📹 Video Upscaler"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video")
                    info_display = gr.Markdown("Upload a video to see metadata.")
                    
                    with gr.Accordion("Video Settings", open=True):
                        v_scale_opt = gr.Radio(["x2", "x4"], label="Upscale Factor", value="x4")
                        v_face_opt = gr.Checkbox(label="Face Enhancement (GFPGAN)", value=True)
                        v_bitrate_opt = gr.Dropdown(["2M", "8M", "20M"], label="Output Bitrate", value="8M")
                        v_rotation_opt = gr.Dropdown(["None", "90° CW", "180°", "90° CCW"], label="Fix Orientation (Manual)", value="None")
                    
                    v_run_btn = gr.Button("🚀 Start Video Upscaling", variant="primary")
                    
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Enhanced Video")
                    v_status_output = gr.Textbox(label="Status")

        # --- Image Tab ---
        with gr.Tab("🖼️ Image Upscaler"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Upload Image", type="numpy")
                    
                    with gr.Accordion("Image Settings", open=True):
                        i_scale_opt = gr.Radio(["x2", "x4"], label="Upscale Factor", value="x4")
                        i_face_opt = gr.Checkbox(label="Face Enhancement (GFPGAN)", value=True)
                        
                        with gr.Group():
                            gr.Markdown("🎨 **Color Correction**")
                            i_color_toggle = gr.Checkbox(label="Enable Color Correction", value=False)
                            i_color_mode = gr.Dropdown(["auto", "vivid", "natural", "low_light"], label="Mode", value="auto")
                        
                        with gr.Group():
                            gr.Markdown("🖼️ **Image Extension (Outpainting)**")
                            i_out_toggle = gr.Checkbox(label="Enable Outpainting", value=False)
                            with gr.Row():
                                i_out_pad = gr.Slider(minimum=0, maximum=500, step=10, label="Extension Padding (Pixels)", value=100)
                            with gr.Accordion("Advanced Resolution Targets", open=False):
                                with gr.Row():
                                    i_out_w = gr.Number(label="Target Width", value=1920)
                                    i_out_h = gr.Number(label="Target Height", value=1080)
                            i_out_mode = gr.Radio(["fast", "ai"], label="Extension Mode", value="fast")
                    
                    i_run_btn = gr.Button("✨ Start Image Enhancement", variant="primary")
                    
                with gr.Column(scale=1):
                    image_output = gr.Image(label="Enhanced Image")
                    i_status_output = gr.Textbox(label="Status")
            
    video_input.change(get_info, inputs=video_input, outputs=info_display)
    v_run_btn.click(process_video, 
                   inputs=[video_input, v_scale_opt, v_face_opt, v_bitrate_opt, v_rotation_opt], 
                   outputs=[video_output, v_status_output])
                   
    i_run_btn.click(process_image,
                   inputs=[image_input, i_scale_opt, i_face_opt, 
                           i_color_toggle, i_color_mode, 
                           i_out_toggle, i_out_w, i_out_h, i_out_pad, i_out_mode],
                   outputs=[image_output, i_status_output])

if __name__ == "__main__":
    demo.launch(share=False)
