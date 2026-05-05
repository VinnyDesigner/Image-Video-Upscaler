import cv2
import numpy as np

def extend_image(image, target_width=None, target_height=None, padding=None, mode="fast"):
    """
    Extend image canvas and fill missing regions intelligently.
    """
    if image is None:
        return None
        
    h, w = image.shape[:2]
    
    # If padding is provided, calculate new target dimensions
    if padding and padding > 0:
        tw = w + (padding * 2)
        th = h + (padding * 2)
    else:
        # Calculate target dimensions (don't shrink)
        tw = max(w, int(target_width)) if target_width else w
        th = max(h, int(target_height)) if target_height else h
    
    if tw <= w and th <= h:
        return image

    # Calculate offsets to center the original image
    x_offset = (tw - w) // 2
    y_offset = (th - h) // 2
    
    if mode == "fast":
        # 1. Fill canvas with background (edge stretching/average)
        # We'll use edge padding for a natural look
        canvas = cv2.copyMakeBorder(
            image, 
            y_offset, th - h - y_offset, 
            x_offset, tw - w - x_offset, 
            cv2.BORDER_REPLICATE
        )
        
        # 2. Smooth the transitions (Blur the edges of the extension)
        mask = np.zeros((th, tw), dtype=np.uint8)
        mask[y_offset:y_offset+h, x_offset:x_offset+w] = 255
        
        # Invert mask to get the "new" areas
        inv_mask = cv2.bitwise_not(mask)
        
        # Apply slight blur to the extended area to hide repetition
        blurred_canvas = cv2.GaussianBlur(canvas, (21, 21), 0)
        canvas = np.where(mask[:, :, None] == 255, canvas, blurred_canvas)
        
        # 3. Inject Noise to match texture
        noise = np.random.normal(0, 5, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 4. Place original image back (sharp)
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = image
        
        # 5. Blend the seams
        seam_mask = cv2.GaussianBlur(mask.astype(float), (15, 15), 0) / 255.0
        seam_mask = seam_mask[:, :, None]
        canvas = (canvas * seam_mask + blurred_canvas * (1 - seam_mask)).astype(np.uint8)
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = image

    elif mode == "ai":
        # Placeholder for AI-based outpainting (e.g. Stable Diffusion)
        # Falling back to fast mode as requested if AI not available
        return extend_image(image, target_width, target_height, mode="fast")

    return canvas
