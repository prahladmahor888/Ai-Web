import cv2
import numpy as np
from PIL import Image
import logging

class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def remove_watermark(self, image, mask_coords):
        """
        Remove watermark using multiple techniques for better results
        """
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Create refined mask
            mask = self._create_refined_mask(img_array, mask_coords)
            
            # Convert to BGR for OpenCV processing
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    # Handle RGBA images
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                else:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Store original color information
            orig_color = img_array.copy()
            
            # Process at multiple scales
            result = self._multi_scale_processing(img_array, mask)
            
            # Preserve original colors
            result = self._preserve_colors(result, orig_color, mask)
            
            # Convert back to RGB
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            return Image.fromarray(result)

        except Exception as e:
            self.logger.error(f"Error in watermark removal: {str(e)}")
            raise

    def _create_refined_mask(self, img, mask_coords):
        """Create a refined mask using image analysis"""
        x, y, w, h = [int(v) for v in mask_coords.values()]
        
        # Create base mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        roi = img[y:y+h, x:x+w]
        
        # Convert ROI to grayscale
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray_roi = roi
            
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Refine mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Update mask with refined region
        mask[y:y+h, x:x+w] = thresh
        return mask

    def _multi_scale_processing(self, img, mask):
        """Process image at multiple scales for better detail preservation"""
        # Generate Gaussian pyramid
        pyramids = [(img, mask)]
        curr_img, curr_mask = img, mask
        
        for _ in range(3):  # Process at 3 different scales
            curr_img = cv2.pyrDown(curr_img)
            curr_mask = cv2.pyrDown(curr_mask)
            pyramids.append((curr_img, curr_mask))
        
        # Process from coarse to fine
        for i, (curr_img, curr_mask) in reversed(list(enumerate(pyramids))):
            curr_result = self._apply_inpainting_techniques(curr_img, curr_mask)
            
            if i < len(pyramids) - 1:
                # Upscale and blend with previous result
                curr_result = cv2.pyrUp(curr_result)
                if curr_result.shape != pyramids[i][0].shape:
                    curr_result = cv2.resize(curr_result, 
                                          (pyramids[i][0].shape[1], pyramids[i][0].shape[0]))
        
        return curr_result

    def _apply_inpainting_techniques(self, img, mask):
        """Enhanced inpainting with multiple passes"""
        # Initial rough inpainting
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        
        # Refined inpainting with smaller radius
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
        
        # Edge-preserving filter with stronger parameters
        result = cv2.edgePreservingFilter(result, flags=2, sigma_s=60, sigma_r=0.4)
        
        # Enhanced detail preservation
        result = cv2.detailEnhance(result, sigma_s=15, sigma_r=0.25)
        
        # Bilateral filtering for texture preservation
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result

    def _preserve_colors(self, result, original, mask):
        """Preserve original colors while keeping new texture"""
        mask_blur = cv2.GaussianBlur(mask.astype(float), (21, 21), 0)
        mask_blur = mask_blur[..., np.newaxis]
        
        # Blend original colors with result
        result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        
        # Keep original color (a,b), but use new luminance (L)
        result_lab[:,:,1:] = orig_lab[:,:,1:]
        
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
