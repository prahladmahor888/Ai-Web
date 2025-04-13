import cv2
import numpy as np
from PIL import Image
import io
import logging
import random

logger = logging.getLogger(__name__)

def enhance_image(image_data, filter_type):
    """
    Apply different AI filters to the image
    """
    try:
        # Convert base64 image to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Ensure image is in RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply initial image enhancement
        img = apply_initial_enhancement(img)

        # Apply selected filter
        if filter_type == 'enhance':
            img = apply_enhancement(img)
        elif filter_type == 'cartoon':
            img = apply_cartoon_effect(img)
        elif filter_type == 'sketch':
            img = apply_sketch_effect(img)
        elif filter_type == 'oil-painting':
            img = apply_oil_painting_effect(img)
        elif filter_type == 'watercolor':
            img = apply_watercolor_effect(img)
        elif filter_type == 'vintage':
            img = apply_vintage_effect(img)
        elif filter_type == 'pop-art':
            img = apply_pop_art_effect(img)
        elif filter_type == 'impressionist':
            img = apply_impressionist_effect(img)
        elif filter_type == 'pointillism':
            img = apply_pointillism_effect(img)
        elif filter_type == 'cubism':
            img = apply_cubism_effect(img)
        elif filter_type == 'mosaic':
            img = apply_mosaic_effect(img)
        elif filter_type == 'stained-glass':
            img = apply_stained_glass_effect(img)
        elif filter_type == 'neon':
            img = apply_neon_effect(img)
        elif filter_type == 'duotone':
            img = apply_duotone_effect(img)
        elif filter_type == 'sepia':
            img = apply_sepia_effect(img)
        elif filter_type == 'infrared':
            img = apply_infrared_effect(img)
        elif filter_type == 'x-ray':
            img = apply_xray_effect(img)
        elif filter_type == 'thermal':
            img = apply_thermal_effect(img)
        elif filter_type == 'paper':
            img = apply_paper_effect(img)
        elif filter_type == 'canvas':
            img = apply_canvas_effect(img)
        elif filter_type == 'metal':
            img = apply_metal_effect(img)
        elif filter_type == 'wood':
            img = apply_wood_effect(img)
        elif filter_type == 'leather':
            img = apply_leather_effect(img)
        elif filter_type == 'fabric':
            img = apply_fabric_effect(img)
        elif filter_type == 'hologram':
            img = apply_hologram_effect(img)
        elif filter_type == 'glitch':
            img = apply_glitch_effect(img)
        elif filter_type == 'pixelate':
            img = apply_pixelate_effect(img)
        elif filter_type == 'ascii':
            img = apply_ascii_effect(img)
        elif filter_type == 'emboss':
            img = apply_emboss_effect(img)
        elif filter_type == 'relief':
            img = apply_relief_effect(img)
        elif filter_type == 'spotlight':
            img = apply_spotlight_effect(img)
        elif filter_type == 'rays':
            img = apply_rays_effect(img)
        elif filter_type == 'lens-flare':
            img = apply_lens_flare_effect(img)
        elif filter_type == 'vignette':
            img = apply_vignette_effect(img)
        elif filter_type == 'halo':
            img = apply_halo_effect(img)
        elif filter_type == 'bloom':
            img = apply_bloom_effect(img)
        elif filter_type == 'rain':
            img = apply_rain_effect(img)
        elif filter_type == 'snow':
            img = apply_snow_effect(img)
        elif filter_type == 'fog':
            img = apply_fog_effect(img)
        elif filter_type == 'mist':
            img = apply_mist_effect(img)
        elif filter_type == 'heat-wave':
            img = apply_heat_wave_effect(img)
        elif filter_type == 'frost':
            img = apply_frost_effect(img)
        elif filter_type == 'day':
            img = apply_day_effect(img)
        elif filter_type == 'night':
            img = apply_night_effect(img)
        elif filter_type == 'sunset':
            img = apply_sunset_effect(img)
        elif filter_type == 'dawn':
            img = apply_dawn_effect(img)
        elif filter_type == 'dusk':
            img = apply_dusk_effect(img)
        elif filter_type == 'moonlight':
            img = apply_moonlight_effect(img)
        elif filter_type == 'cinematic':
            img = apply_cinematic_effect(img)
        elif filter_type == 'drama':
            img = apply_drama_effect(img)
        elif filter_type == 'noir':
            img = apply_noir_effect(img)
        elif filter_type == 'silver-screen':
            img = apply_silver_screen_effect(img)
        elif filter_type == 'technicolor':
            img = apply_technicolor_effect(img)
        elif filter_type == 'kodachrome':
            img = apply_kodachrome_effect(img)
        elif filter_type == 'portrait':
            img = apply_portrait_effect(img)
        elif filter_type == 'beauty':
            img = apply_beauty_effect(img)
        elif filter_type == 'glamour':
            img = apply_glamour_effect(img)
        elif filter_type == 'vintage-portrait':
            img = apply_vintage_portrait_effect(img)
        elif filter_type == 'high-key':
            img = apply_high_key_effect(img)
        elif filter_type == 'low-key':
            img = apply_low_key_effect(img)
        elif filter_type == 'landscape':
            img = apply_landscape_effect(img)
        elif filter_type == 'hdr-landscape':
            img = apply_hdr_landscape_effect(img)
        elif filter_type == 'golden-hour':
            img = apply_golden_hour_effect(img)
        elif filter_type == 'blue-hour':
            img = apply_blue_hour_effect(img)
        elif filter_type == 'dramatic-sky':
            img = apply_dramatic_sky_effect(img)
        elif filter_type == 'autumn':
            img = apply_autumn_effect(img)
        elif filter_type == 'teal-orange':
            img = apply_teal_orange_effect(img)
        elif filter_type == 'cool-blue':
            img = apply_cool_blue_effect(img)
        elif filter_type == 'warm-gold':
            img = apply_warm_gold_effect(img)
        elif filter_type == 'muted':
            img = apply_muted_effect(img)
        elif filter_type == 'vibrant':
            img = apply_vibrant_effect(img)
        elif filter_type == 'pastel':
            img = apply_pastel_effect(img)
        elif filter_type == 'acrylic':
            img = apply_acrylic_effect(img)
        elif filter_type == 'charcoal':
            img = apply_charcoal_effect(img)
        elif filter_type == 'ink':
            img = apply_ink_effect(img)
        elif filter_type == 'pencil':
            img = apply_pencil_effect(img)
        elif filter_type == 'double-exposure':
            img = apply_double_exposure_effect(img)
        elif filter_type == 'tilt-shift':
            img = apply_tilt_shift_effect(img)
        elif filter_type == 'cross-process':
            img = apply_cross_process_effect(img)
        elif filter_type == 'lomo':
            img = apply_lomo_effect(img)
        elif filter_type == 'polaroid':
            img = apply_polaroid_effect(img)
        elif filter_type == 'holga':
            img = apply_holga_effect(img)
        elif filter_type == 'melancholic':
            img = apply_melancholic_effect(img)
        elif filter_type == 'nostalgic':
            img = apply_nostalgic_effect(img)
        elif filter_type == 'romantic':
            img = apply_romantic_effect(img)
        elif filter_type == 'mysterious':
            img = apply_mysterious_effect(img)
        elif filter_type == 'dreamy':
            img = apply_dreamy_effect(img)
        elif filter_type == 'dramatic':
            img = apply_dramatic_effect(img)
        elif filter_type == 'instagram':
            img = apply_instagram_effect(img)
        elif filter_type == 'vintage-modern':
            img = apply_vintage_modern_effect(img)
        elif filter_type == 'minimalist':
            img = apply_minimalist_effect(img)
        elif filter_type == 'urban':
            img = apply_urban_effect(img)
        elif filter_type == 'cyberpunk':
            img = apply_cyberpunk_effect(img)
        elif filter_type == 'retrowave':
            img = apply_retrowave_effect(img)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        # Apply final image enhancement
        img = apply_final_enhancement(img)

        # Convert back to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = buffer.tobytes()
        
        return img_base64

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def apply_initial_enhancement(img):
    """Apply initial image enhancement"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Enhance color channels
    a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    
    # Merge channels
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Apply slight sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img

def apply_final_enhancement(img):
    """Apply final image enhancement"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Enhance color channels
    a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    
    # Merge channels
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Apply slight sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img

def apply_background_blur(img, strength=0.7):
    """Apply smooth background blur while preserving edges"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter for edge-preserving blur
    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Create edge mask
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
    
    # Create smooth mask
    mask = cv2.GaussianBlur(edges.astype(float), (5,5), 0)
    
    # Blend original and blurred image
    result = cv2.addWeighted(img, 1, blurred, strength, 0)
    
    return result

def apply_enhancement(img):
    """Apply high-quality image enhancement"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply background blur
    img = apply_background_blur(img, 0.5)
    
    # Apply color enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Enhance colors
    a = cv2.multiply(a, 1.1)
    b = cv2.multiply(b, 1.1)
    
    # Merge channels
    enhanced_lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return img

def apply_cartoon_effect(img):
    """Apply high-quality cartoon effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter for edge preservation
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Detect edges with improved parameters
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply color quantization with improved parameters
    img = cv2.medianBlur(img, 5)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Apply color reduction
    img = cv2.pyrMeanShiftFiltering(img, 20, 30)
    
    # Combine edges with color
    img = cv2.bitwise_and(img, img, mask=edges)
    
    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    return img

def apply_sketch_effect(img):
    """Apply high-quality sketch effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur with improved parameters
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Create sketch effect with improved blending
    sketch = cv2.divide(gray, blurred, scale=256.0)
    
    # Apply bilateral filter for smoothness
    sketch = cv2.bilateralFilter(sketch, 9, 75, 75)
    
    # Enhance edges
    edges = cv2.Canny(sketch, 100, 200)
    sketch = cv2.addWeighted(sketch, 0.7, edges, 0.3, 0)
    
    # Convert back to RGB with improved contrast
    img = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    
    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    return img

def apply_oil_painting_effect(img):
    """Apply high-quality oil painting effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply bilateral filter for color smoothing
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Apply color quantization with improved parameters
    img = cv2.medianBlur(img, 7)
    img = cv2.pyrMeanShiftFiltering(img, 20, 30)
    
    # Apply edge enhancement
    edges = cv2.Canny(img, 100, 200)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
    
    # Apply texture with improved parameters
    texture = np.random.normal(0, 8, img.shape).astype(np.uint8)
    img = cv2.add(img, texture)
    
    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    return img

def apply_watercolor_effect(img):
    """Apply high-quality watercolor effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply bilateral filter for color smoothing
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Apply color quantization with improved parameters
    img = cv2.medianBlur(img, 7)
    img = cv2.pyrMeanShiftFiltering(img, 20, 30)
    
    # Apply edge enhancement
    edges = cv2.Canny(img, 100, 200)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
    
    # Apply texture with improved parameters
    texture = np.random.normal(0, 4, img.shape).astype(np.uint8)
    img = cv2.add(img, texture)
    
    # Apply watercolor texture
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    
    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    return img

def apply_vintage_effect(img):
    """Apply high-quality vintage effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply sepia effect with improved parameters
    l = cv2.multiply(l, 0.85)  # Slightly darker
    a = cv2.multiply(a, 0.75)  # Reduced red
    b = cv2.multiply(b, 0.65)  # Reduced blue
    
    # Merge channels
    vintage_lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(vintage_lab, cv2.COLOR_LAB2RGB)
    
    # Apply vignette with improved parameters
    height, width = img.shape[:2]
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    vignette = np.sqrt(xx**2 + yy**2)
    vignette = (vignette - vignette.min()) / (vignette.max() - vignette.min())
    vignette = cv2.GaussianBlur(vignette, (31,31), 0)  # Larger blur
    vignette = vignette.reshape(height, width, 1)
    img = img * (1 - vignette * 0.6)  # Stronger vignette
    
    # Add slight noise
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_pop_art_effect(img):
    """Apply high-quality pop art effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply posterization with improved parameters
    _, poster = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Apply color quantization
    img = cv2.pyrMeanShiftFiltering(img, 20, 30)
    
    # Combine with posterized edges
    edges = cv2.Canny(poster, 100, 200)
    img = cv2.bitwise_and(img, img, mask=edges)
    
    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l,a,b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    
    return img

def apply_impressionist_effect(img):
    """Apply high-quality impressionist effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    
    # Apply edge enhancement
    edges = cv2.Canny(blurred, 100, 200)
    
    # Combine blurred and edge-enhanced images
    img = cv2.addWeighted(blurred, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.3, 0)
    
    return img

def apply_pointillism_effect(img):
    """Apply high-quality pointillism effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Create pointillism effect
    height, width = img.shape[:2]
    result = np.zeros_like(img)
    for _ in range(10000):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        color = img[y, x]
        cv2.circle(result, (x, y), 2, color.tolist(), -1)
    
    return result

def apply_cubism_effect(img):
    """Apply high-quality cubism effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply edge enhancement
    edges = cv2.Canny(gray, 100, 200)
    
    # Combine blurred and edge-enhanced images
    img = cv2.addWeighted(img, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.3, 0)
    
    return img

def apply_mosaic_effect(img):
    """Apply high-quality mosaic effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Create mosaic effect
    height, width = img.shape[:2]
    tile_size = 10
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size]
            if tile.size > 0:
                color = np.mean(tile, axis=(0,1))
                img[y:y+tile_size, x:x+tile_size] = color
    
    return img

def apply_stained_glass_effect(img):
    """Apply high-quality stained glass effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Combine original and binary images
    img = cv2.addWeighted(img, 0.7, cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), 0.3, 0)
    
    return img

def apply_neon_effect(img):
    """Apply high-quality neon effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    
    # Apply color enhancement
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    
    return img

def apply_duotone_effect(img):
    """Apply high-quality duotone effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply color quantization
    img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    return img

def apply_sepia_effect(img):
    """Apply high-quality sepia effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply sepia effect
    kernel = np.array([[0.393, 0.769, 0.189],
                     [0.349, 0.686, 0.168],
                     [0.272, 0.534, 0.131]])
    img = cv2.transform(img, kernel)
    
    return img

def apply_infrared_effect(img):
    """Apply high-quality infrared effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply color quantization
    img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    return img

def apply_xray_effect(img):
    """Apply high-quality X-ray effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply X-ray effect
    img = cv2.bitwise_not(gray)
    
    return img

def apply_thermal_effect(img):
    """Apply high-quality thermal effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply color quantization
    img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    return img

def apply_paper_effect(img):
    """Apply high-quality paper texture effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply noise
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_canvas_effect(img):
    """Apply high-quality canvas texture effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply noise
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_metal_effect(img):
    """Apply high-quality metallic effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply color quantization
    img = cv2.applyColorMap(gray, cv2.COLORMAP_COOL)
    
    return img

def apply_wood_effect(img):
    """Apply high-quality wood texture effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply noise
    noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_leather_effect(img):
    """Apply high-quality leather texture effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply noise
    noise = np.random.normal(0, 30, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_fabric_effect(img):
    """Apply high-quality fabric texture effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply noise
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_hologram_effect(img):
    """Apply high-quality hologram effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    
    # Apply hologram effect
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    
    return img

def apply_glitch_effect(img):
    """Apply high-quality glitch effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply glitch effect
    height, width = img.shape[:2]
    shift = random.randint(-50, 50)
    img[:, shift:] = img[:, :-shift]
    
    return img

def apply_pixelate_effect(img):
    """Apply high-quality pixelation effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply pixelation effect
    height, width = img.shape[:2]
    temp = cv2.resize(img, (width//10, height//10), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return img

def apply_ascii_effect(img):
    """Apply high-quality ASCII art effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply ASCII art effect
    img = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2), interpolation=cv2.INTER_NEAREST)
    
    return img

def apply_emboss_effect(img):
    """Apply high-quality emboss effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply emboss effect
    kernel = np.array([[-2,-1,0],
                     [-1, 1,1],
                     [ 0, 1,2]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img

def apply_relief_effect(img):
    """Apply high-quality relief effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply relief effect
    kernel = np.array([[-1,-1,-1],
                     [-1, 9,-1],
                     [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img

def apply_spotlight_effect(img):
    """Apply high-quality spotlight effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply spotlight effect
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width//2, height//2)
    radius = min(width, height)//3
    cv2.circle(mask, center, radius, 255, -1)
    img = cv2.addWeighted(img, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.3, 0)
    
    return img

def apply_rays_effect(img):
    """Apply high-quality light rays effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply light rays effect
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(8):
        angle = i * 45
        cv2.line(mask, (width//2, height//2), 
                (int(width//2 + 100*np.cos(np.radians(angle))),
                 int(height//2 + 100*np.sin(np.radians(angle)))), 255, 2)
    img = cv2.addWeighted(img, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.3, 0)
    
    return img

def apply_lens_flare_effect(img):
    """Apply high-quality lens flare effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply lens flare effect
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width//2, height//2)
    cv2.circle(mask, center, 50, 255, -1)
    img = cv2.addWeighted(img, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.3, 0)
    
    return img

def apply_vignette_effect(img):
    """Apply high-quality vignette effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply vignette effect
    height, width = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(width, 30)
    kernel_y = cv2.getGaussianKernel(height, 30)
    kernel = kernel_y * kernel_x.T
    mask = kernel / np.linalg.norm(kernel)
    img = img * mask
    
    return img

def apply_halo_effect(img):
    """Apply high-quality halo effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply halo effect
    blurred = cv2.GaussianBlur(img, (21,21), 0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    
    return img

def apply_bloom_effect(img):
    """Apply high-quality bloom effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    
    # Apply bloom effect
    img = cv2.addWeighted(img, 1.2, blurred, 0.3, 0)
    
    return img

def apply_rain_effect(img):
    """Apply high-quality rain effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply noise
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_snow_effect(img):
    """Apply high-quality snow effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply noise
    noise = np.random.normal(0, 30, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_fog_effect(img):
    """Apply high-quality fog effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (15,15), 0)
    
    # Apply fog effect
    img = cv2.addWeighted(img, 0.7, blurred, 0.3, 0)
    
    return img

def apply_mist_effect(img):
    """Apply high-quality mist effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (10,10), 0)
    
    # Apply mist effect
    img = cv2.addWeighted(img, 0.8, blurred, 0.2, 0)
    
    return img

def apply_heat_wave_effect(img):
    """Apply high-quality heat wave effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply heat wave effect
    height, width = img.shape[:2]
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    x_offset = np.sin(y_coords/30) * 10
    x_coords = x_coords + x_offset.astype(np.int32)
    x_coords = np.clip(x_coords, 0, width-1)
    img = img[y_coords, x_coords]
    
    return img

def apply_frost_effect(img):
    """Apply high-quality frost effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply noise
    noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_day_effect(img):
    """Apply high-quality bright daylight effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply day effect
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    
    return img

def apply_night_effect(img):
    """Apply high-quality night time effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply night effect
    img = cv2.convertScaleAbs(img, alpha=0.5, beta=-30)
    
    return img

def apply_sunset_effect(img):
    """Apply high-quality warm sunset effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply sunset effect
    kernel = np.array([[1.2, 0, 0],
                     [0, 1.0, 0],
                     [0, 0, 0.8]])
    img = cv2.transform(img, kernel)
    
    return img

def apply_dawn_effect(img):
    """Apply high-quality early morning effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply dawn effect
    img = cv2.convertScaleAbs(img, alpha=0.8, beta=20)
    
    return img

def apply_dusk_effect(img):
    """Apply high-quality evening effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply dusk effect
    img = cv2.convertScaleAbs(img, alpha=0.7, beta=-10)
    
    return img

def apply_moonlight_effect(img):
    """Apply high-quality moonlight effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply moonlight effect
    kernel = np.array([[0.8, 0, 0],
                     [0, 0.8, 0],
                     [0, 0, 1.2]])
    img = cv2.transform(img, kernel)
    
    return img

def apply_cinematic_effect(img):
    """Apply high-quality cinematic effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply cinematic effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply teal and orange color grading
    a = cv2.multiply(a, 1.2)  # Enhance orange
    b = cv2.multiply(b, 0.8)  # Enhance teal
    
    # Merge channels
    cinematic_lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(cinematic_lab, cv2.COLOR_LAB2RGB)
    
    # Apply vignette
    height, width = img.shape[:2]
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    vignette = np.sqrt(xx**2 + yy**2)
    vignette = (vignette - vignette.min()) / (vignette.max() - vignette.min())
    vignette = cv2.GaussianBlur(vignette, (21,21), 0)
    vignette = vignette.reshape(height, width, 1)
    img = img * (1 - vignette * 0.3)
    
    return img

def apply_drama_effect(img):
    """Apply high-quality dramatic effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply dramatic effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_noir_effect(img):
    """Apply high-quality film noir effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply film noir effect
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    img = cv2.addWeighted(img, 0.7, cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), 0.3, 0)
    
    return img

def apply_silver_screen_effect(img):
    """Apply high-quality classic black and white movie effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply silver screen effect
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return img

def apply_technicolor_effect(img):
    """Apply high-quality vintage technicolor effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply technicolor effect
    b, g, r = cv2.split(img)
    r = cv2.add(r, 30)  # Boost reds
    g = cv2.add(g, 20)  # Boost greens
    b = cv2.add(b, 10)  # Boost blues
    img = cv2.merge((b,g,r))
    
    return img

def apply_kodachrome_effect(img):
    """Apply high-quality Kodachrome film effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Kodachrome effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, 10)  # Brighten
    a = cv2.add(a, 5)   # Slight red boost
    b = cv2.add(b, -5)  # Slight blue reduction
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_portrait_effect(img):
    """Apply high-quality portrait effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply background blur
    img = apply_background_blur(img, 0.7)
    
    # Apply portrait effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_beauty_effect(img):
    """Apply high-quality beauty effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply bilateral filter for skin smoothing
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Apply portrait effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_glamour_effect(img):
    """Apply high-quality glamorous effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply glamour effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, 15)  # Brighten
    a = cv2.add(a, 10)  # Boost reds
    b = cv2.add(b, -5)  # Reduce blues
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_vintage_portrait_effect(img):
    """Apply high-quality vintage portrait effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply sepia effect
    sepia = cv2.transform(img, np.array([[0.393, 0.769, 0.189],
                                       [0.349, 0.686, 0.168],
                                       [0.272, 0.534, 0.131]]))
    img = cv2.addWeighted(img, 0.7, sepia, 0.3, 0)
    
    return img

def apply_high_key_effect(img):
    """Apply high-quality high-key portrait effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply high-key effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, 30)  # Brighten significantly
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_low_key_effect(img):
    """Apply high-quality low-key portrait effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply low-key effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, -30)  # Darken significantly
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_landscape_effect(img):
    """Apply high-quality landscape effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply landscape effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Apply HDR effect
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    
    return img

def apply_hdr_landscape_effect(img):
    """Apply high-quality HDR landscape effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply HDR effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Apply HDR effect
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=15)
    
    return img

def apply_golden_hour_effect(img):
    """Apply high-quality golden hour effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply golden hour effect
    b, g, r = cv2.split(img)
    r = cv2.add(r, 30)  # Boost reds
    g = cv2.add(g, 20)  # Boost greens
    b = cv2.add(b, -10)  # Reduce blues
    img = cv2.merge((b,g,r))
    
    return img

def apply_blue_hour_effect(img):
    """Apply high-quality blue hour effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply blue hour effect
    b, g, r = cv2.split(img)
    r = cv2.add(r, -10)  # Reduce reds
    g = cv2.add(g, -5)   # Reduce greens
    b = cv2.add(b, 20)   # Boost blues
    img = cv2.merge((b,g,r))
    
    return img

def apply_dramatic_sky_effect(img):
    """Apply high-quality dramatic sky effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply dramatic sky effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_autumn_effect(img):
    """Apply high-quality autumn effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply autumn effect
    b, g, r = cv2.split(img)
    r = cv2.add(r, 20)  # Boost reds
    g = cv2.add(g, 10)  # Slight green boost
    b = cv2.add(b, -10)  # Reduce blues
    img = cv2.merge((b,g,r))
    
    return img

def apply_teal_orange_effect(img):
    """Apply high-quality teal and orange effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply teal and orange effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.multiply(a, 1.3)  # Enhance orange
    b = cv2.multiply(b, 0.7)  # Enhance teal
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_cool_blue_effect(img):
    """Apply high-quality cool blue effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply cool blue effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.multiply(a, 0.8)  # Reduce orange
    b = cv2.multiply(b, 1.2)  # Enhance blue
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_warm_gold_effect(img):
    """Apply high-quality warm gold effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply warm gold effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.multiply(a, 1.2)  # Enhance orange
    b = cv2.multiply(b, 1.1)  # Enhance yellow
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_muted_effect(img):
    """Apply high-quality muted effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply muted effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.multiply(a, 0.8)  # Reduce orange
    b = cv2.multiply(b, 0.8)  # Reduce blue
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_vibrant_effect(img):
    """Apply high-quality vibrant effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply vibrant effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.multiply(a, 1.3)  # Enhance orange
    b = cv2.multiply(b, 1.3)  # Enhance blue
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_pastel_effect(img):
    """Apply high-quality pastel effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply pastel effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.multiply(a, 0.7)  # Reduce orange
    b = cv2.multiply(b, 0.7)  # Reduce blue
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_acrylic_effect(img):
    """Apply high-quality acrylic painting effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply acrylic effect
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    edges = cv2.Canny(bilateral, 100, 200)
    img = cv2.addWeighted(bilateral, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.3, 0)
    
    return img

def apply_charcoal_effect(img):
    """Apply high-quality charcoal drawing effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply charcoal effect
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return img

def apply_ink_effect(img):
    """Apply high-quality ink drawing effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply ink effect
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    return img

def apply_pencil_effect(img):
    """Apply high-quality pencil sketch effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply pencil effect
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - gray
    blurred = cv2.GaussianBlur(inv_gray, (21,21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
    img = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    
    return img

def apply_double_exposure_effect(img):
    """Apply high-quality double exposure effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply double exposure effect
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.addWeighted(img, 0.7, blurred, 0.3, 0)
    
    return img

def apply_tilt_shift_effect(img):
    """Apply high-quality tilt-shift effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply tilt-shift effect
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width//2, height//2)
    radius = min(width, height)//3
    cv2.circle(mask, center, radius, 255, -1)
    blurred = cv2.GaussianBlur(img, (15,15), 0)
    img = cv2.addWeighted(img, 1, blurred, 0.5, 0)
    
    return img

def apply_cross_process_effect(img):
    """Apply high-quality cross-processed film effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply cross-process effect
    b, g, r = cv2.split(img)
    r = cv2.add(r, 30)  # Boost reds
    g = cv2.add(g, -10)  # Reduce greens
    b = cv2.add(b, -20)  # Reduce blues
    img = cv2.merge((b,g,r))
    
    return img

def apply_lomo_effect(img):
    """Apply high-quality Lomo effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Lomo effect
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_polaroid_effect(img):
    """Apply high-quality Polaroid effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Polaroid effect
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_holga_effect(img):
    """Apply high-quality Holga effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Holga effect
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def apply_melancholic_effect(img):
    """Apply high-quality melancholic effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply melancholic effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, -10)  # Darken
    a = cv2.add(a, -10)  # Reduce red-green
    b = cv2.add(b, -10)  # Reduce blue-yellow
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_nostalgic_effect(img):
    """Apply high-quality nostalgic effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply nostalgic effect
    sepia = cv2.transform(img, np.array([[0.393, 0.769, 0.189],
                                       [0.349, 0.686, 0.168],
                                       [0.272, 0.534, 0.131]]))
    img = cv2.addWeighted(img, 0.7, sepia, 0.3, 0)
    
    return img

def apply_romantic_effect(img):
    """Apply high-quality romantic effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply romantic effect
    b, g, r = cv2.split(img)
    r = cv2.add(r, 20)  # Boost reds
    g = cv2.add(g, 10)  # Boost greens
    b = cv2.add(b, -10)  # Reduce blues
    img = cv2.merge((b,g,r))
    
    return img

def apply_mysterious_effect(img):
    """Apply high-quality mysterious effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply mysterious effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, -20)  # Darken
    a = cv2.add(a, -15)  # Reduce red-green
    b = cv2.add(b, -15)  # Reduce blue-yellow
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_dreamy_effect(img):
    """Apply high-quality dreamy effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply dreamy effect
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.addWeighted(img, 0.7, blurred, 0.3, 0)
    
    return img

def apply_dramatic_effect(img):
    """Apply high-quality dramatic effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply dramatic effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_instagram_effect(img):
    """Apply high-quality Instagram effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply Instagram effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, 10)  # Brighten
    a = cv2.add(a, 5)   # Slight red boost
    b = cv2.add(b, -5)  # Slight blue reduction
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_vintage_modern_effect(img):
    """Apply high-quality vintage-modern blend"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply vintage-modern blend
    sepia = cv2.transform(img, np.array([[0.393, 0.769, 0.189],
                                       [0.349, 0.686, 0.168],
                                       [0.272, 0.534, 0.131]]))
    img = cv2.addWeighted(img, 0.8, sepia, 0.2, 0)
    
    return img

def apply_minimalist_effect(img):
    """Apply high-quality minimalist effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply minimalist effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.add(a, -10)  # Reduce red-green
    b = cv2.add(b, -10)  # Reduce blue-yellow
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_urban_effect(img):
    """Apply high-quality urban effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply urban effect
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return img

def apply_cyberpunk_effect(img):
    """Apply high-quality cyberpunk effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply cyberpunk effect
    b, g, r = cv2.split(img)
    r = cv2.add(r, 30)  # Boost reds
    g = cv2.add(g, -10)  # Reduce greens
    b = cv2.add(b, 30)   # Boost blues
    img = cv2.merge((b,g,r))
    
    return img

def apply_retrowave_effect(img):
    """Apply high-quality retrowave effect"""
    # Apply initial enhancement
    img = apply_initial_enhancement(img)
    
    # Apply retrowave effect
    b, g, r = cv2.split(img)
    r = cv2.add(r, 20)  # Boost reds
    g = cv2.add(g, -10)  # Reduce greens
    b = cv2.add(b, 20)   # Boost blues
    img = cv2.merge((b,g,r))
    
    return img

def encode_image(image):
    """
    Encode numpy array image to bytes
    """
    try:
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise 