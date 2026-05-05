import cv2
import numpy as np

def apply_color_correction(image, mode="auto"):
    """
    Enhance image colors using LAB color space to preserve natural tones.
    """
    if image is None:
        return None

    # Convert to LAB to work on Luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if mode == "auto":
        # White Balance (Simple Gray World)
        avg_a = np.average(a)
        avg_b = np.average(b)
        a = np.clip(a - ((avg_a - 128) * (l / 255.0) * 1.1), 0, 255).astype(np.uint8)
        b = np.clip(b - ((avg_b - 128) * (l / 255.0) * 1.1), 0, 255).astype(np.uint8)
        
        # Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

    elif mode == "vivid":
        # Increase Saturation
        a = cv2.addWeighted(a, 1.2, np.full(a.shape, 128, dtype=np.uint8), -0.2, 0)
        b = cv2.addWeighted(b, 1.2, np.full(b.shape, 128, dtype=np.uint8), -0.2, 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

    elif mode == "low_light":
        # Gamma correction for shadows
        invGamma = 1.0 / 1.5
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        l = cv2.LUT(l, table)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)

    elif mode == "natural":
        # Subtle contrast only
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l = clahe.apply(l)

    # Merge channels and convert back
    l_final = cv2.merge((l, a, b))
    result = cv2.cvtColor(l_final, cv2.COLOR_LAB2BGR)

    # Subtle Sharpening (Normalized to preserve brightness)
    # The kernel should sum to 1.0 to maintain the same average intensity.
    sharpen_amount = 0.15 # Adjust this for more/less sharpening
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ]) * (sharpen_amount / 9.0)
    kernel[1, 1] = 1.0 - (np.sum(kernel) - kernel[1, 1])
    
    result = cv2.filter2D(result, -1, kernel)

    return result
