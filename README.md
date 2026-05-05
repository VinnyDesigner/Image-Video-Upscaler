# 🎥 AI Video Upscaler Pro

AI Video Upscaler Pro is a production-grade Python application designed to convert low-resolution videos into high-quality 4K enhanced videos using state-of-the-art AI super-resolution and face enhancement models.

## 🚀 Features

- **AI Super-Resolution:** Powered by Real-ESRGAN for sharp, clean upscaling (2x, 4x).
- **Face Enhancement:** Integrated GFPGAN for restoring facial details and textures.
- **Production-Ready Pipeline:** 
  - Frame-by-frame processing with FFmpeg integration.
  - Automatic audio synchronization.
  - Bitrate and resolution control.
- **Modern UI:** Responsive dark-themed dashboard built with Gradio.
- **Hardware Acceleration:** Supports NVIDIA GPU (CUDA) with automatic CPU fallback.

## 🛠️ Tech Stack

- **Core:** Python 3.10+
- **Deep Learning:** PyTorch, Real-ESRGAN, GFPGAN
- **Video Handling:** OpenCV, FFmpeg
- **UI:** Gradio

## 📦 Installation

1. **Install FFmpeg:**
   Ensure FFmpeg is installed on your system and added to your PATH.
   - [Download FFmpeg](https://ffmpeg.org/download.html)

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python main.py
   ```

## 📂 Project Structure

- `main.py`: Entry point for the application.
- `ui.py`: Gradio-based web interface.
- `pipeline.py`: Main processing logic coordinating all steps.
- `upscaler.py`: AI super-resolution logic.
- `face_enhancer.py`: Facial restoration logic.
- `video_writer.py`: Video reassembly and audio merging.
- `frame_extractor.py`: Frame extraction from source video.
- `config.py`: Application settings and constants.
- `utils.py`: Helper functions and model management.

## 💡 Usage

1. Upload your video (MP4, MOV, AVI).
2. Select the upscale factor (x2 or x4).
3. Enable "Face Enhancement" for portraits or close-ups.
4. Click **Start Upscaling**.
5. Download the processed video from the output section.

## ⚖️ License

MIT License
