import cv2
import numpy as np
from PIL import Image

class WatermarkDetector:
    def detect(self, image):
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find potential watermark regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and get the most likely watermark region
        if contours:
            # Sort contours by area, get the largest
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Convert coordinates to relative positions
            return {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        
        return None
