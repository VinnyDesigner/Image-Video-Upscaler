import cv2
import numpy as np
import time
import uuid
import os
from pathlib import Path
from config import OUTPUTS_DIR
from upscaler import AIUpscaler
from PIL import Image, ImageDraw, ImageFont

class OCRPipeline:
    def __init__(self):
        self.easyocr_reader = None
        self.paddleocr_reader = None
        
    def _apply_screenshot_mode(self, img):
        # CLAHE, Bilateral Filtering, Local Contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # Bilateral filter for edge preservation
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return enhanced

    def _apply_text_super_resolution(self, img_region, strength):
        scale = 1
        if strength == "Medium": scale = 2
        elif strength == "High": scale = 4
        
        enhanced = img_region.copy()
        if scale > 1:
            try:
                upscaler = AIUpscaler(scale=scale)
                enhanced = upscaler.upscale(enhanced)
            except Exception as e:
                print(f"Upscaling failed: {e}")
                
        # Adaptive thresholding & morphological sharpening
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Sharpening via custom kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Resize back to original region bounds to reinsert
        h, w = img_region.shape[:2]
        enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_AREA)
        return enhanced

    def get_ocr_results(self, img, engine):
        results = [] # list of dicts: {'bbox': (x1, y1, x2, y2), 'text': str, 'prob': float}
        if engine in ["EasyOCR", "Auto"]:
            if self.easyocr_reader is None:
                try:
                    import easyocr
                    self.easyocr_reader = easyocr.Reader(['en'], gpu=True)
                except ImportError:
                    return results, "EasyOCR not installed."
            
            res = self.easyocr_reader.readtext(img)
            for (bbox, text, prob) in res:
                tl, tr, br, bl = bbox
                x1, y1 = int(min(tl[0], bl[0])), int(min(tl[1], tr[1]))
                x2, y2 = int(max(tr[0], br[0])), int(max(bl[1], br[1]))
                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                results.append({'bbox': (x1, y1, x2, y2), 'text': text, 'prob': prob})
            return results, f"EasyOCR: {len(results)} regions."
            
        elif engine == "PaddleOCR":
            if self.paddleocr_reader is None:
                try:
                    from paddleocr import PaddleOCR
                    self.paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                except ImportError:
                    return results, "PaddleOCR not installed."
            res = self.paddleocr_reader.ocr(img, cls=True)
            if res and res[0]:
                for line in res[0]:
                    bbox = line[0]
                    text = line[1][0]
                    prob = line[1][1]
                    tl, tr, br, bl = bbox
                    x1, y1 = int(min(tl[0], bl[0])), int(min(tl[1], tr[1]))
                    x2, y2 = int(max(tr[0], br[0])), int(max(bl[1], br[1]))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    results.append({'bbox': (x1, y1, x2, y2), 'text': text, 'prob': prob})
            return results, f"PaddleOCR: {len(results)} regions."
        
        return results, "No OCR performed."

    def apply_ai_reconstruction(self, img, ocr_results):
        # Draw background color over bbox and rewrite text
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            # Attempt to use Arial or default font
            if os.name == 'nt':
                font = ImageFont.truetype("arial.ttf", 16)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            
        for res in ocr_results:
            x1, y1, x2, y2 = res['bbox']
            text = res['text']
            
            # Sample background color near the text
            bg_color = (255, 255, 255)
            if y1 > 2 and x1 > 2:
                bg_color = tuple(img[y1-2, x1-2].tolist()[::-1]) # BGR to RGB
            
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)
            
            # Estimate text color (invert bg roughly)
            brightness = sum(bg_color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
            
            draw.text((x1, y1), text, fill=text_color, font=font)
            
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def process(self, image, engine="Auto", recovery_opts=None, strength="Medium", output_mode="Visual Enhancement Only", ai_reconstruct=False):
        if recovery_opts is None:
            recovery_opts = []
            
        start_time = time.time()
        base_img = image.copy()
        
        # 1. OCR Detection Pre-pass
        ocr_results = []
        ocr_msg = ""
        
        needs_ocr = "Text Detection" in recovery_opts or output_mode in ["OCR Overlay", "Clean Reconstructed Screenshot", "Confidence Heatmap"] or ai_reconstruct
        
        if needs_ocr:
            ocr_results, ocr_msg = self.get_ocr_results(base_img, engine)
            
        # 2. Screenshot Mode & Standard Enhancement (Non-Text / Base)
        if "Screenshot Mode" in recovery_opts:
            base_img = self._apply_screenshot_mode(base_img)
            
        # 3. Text Region Super Resolution
        final_img = base_img.copy()
        if ocr_results and "Text Restoration" in recovery_opts:
            for res in ocr_results:
                x1, y1, x2, y2 = res['bbox']
                if x2 <= x1 or y2 <= y1: continue
                region = final_img[y1:y2, x1:x2]
                enhanced_region = self._apply_text_super_resolution(region, strength)
                final_img[y1:y2, x1:x2] = enhanced_region
                
        # 4. Output Modes & Heatmap
        if output_mode == "Confidence Heatmap" and ocr_results:
            overlay = final_img.copy()
            for res in ocr_results:
                x1, y1, x2, y2 = res['bbox']
                prob = res['prob']
                color = (0, 0, 255) if prob < 0.5 else ((0, 255, 255) if prob < 0.8 else (0, 255, 0))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.4, final_img, 0.6, 0, final_img)
        elif output_mode == "OCR Overlay":
            for res in ocr_results:
                x1, y1, x2, y2 = res['bbox']
                cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # 5. AI Reconstruction
        if ai_reconstruct and ocr_results:
            final_img = self.apply_ai_reconstruction(final_img, ocr_results)
            
        # 6. Generate Exports
        session_id = str(uuid.uuid4())[:8]
        png_path = str(OUTPUTS_DIR / f"ocr_{session_id}.png")
        jpg_path = str(OUTPUTS_DIR / f"ocr_{session_id}.jpg")
        txt_path = str(OUTPUTS_DIR / f"ocr_{session_id}.txt")
        pdf_path = str(OUTPUTS_DIR / f"ocr_{session_id}.pdf")
        
        cv2.imwrite(png_path, final_img)
        cv2.imwrite(jpg_path, final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Save TXT
        with open(txt_path, "w", encoding="utf-8") as f:
            for res in ocr_results:
                f.write(f"{res['text']}\n")
                
        # Save Searchable PDF
        try:
            from reportlab.pdfgen import canvas
            h, w = final_img.shape[:2]
            c = canvas.Canvas(pdf_path, pagesize=(w, h))
            # Draw image
            c.drawImage(png_path, 0, 0, width=w, height=h)
            # Draw invisible text
            for res in ocr_results:
                x1, y1, x2, y2 = res['bbox']
                text = res['text']
                # reportlab origin is bottom-left, so we flip y
                pdf_y = h - y2
                c.saveState()
                c.setFillAlpha(0) # Invisible text
                font_size = y2 - y1 if (y2 - y1) > 0 else 12
                c.setFont("Helvetica", font_size)
                c.drawString(x1, pdf_y, text)
                c.restoreState()
            c.save()
            files_to_return = [png_path, jpg_path, txt_path, pdf_path]
        except Exception as e:
            print(f"PDF export failed: {e}")
            files_to_return = [png_path, jpg_path, txt_path]
            
        duration = time.time() - start_time
        status = f"OCR Processing complete in {duration:.2f}s. {ocr_msg}"
        
        return final_img, status, files_to_return
