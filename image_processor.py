import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    """Image preprocessing and augmentation utilities"""
    
    @staticmethod
    def resize_image(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = img.shape[:2]
        
        if max(height, width) <= max_size:
            return img
        
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize image brightness and contrast"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    @staticmethod
    def denoise_image(img: np.ndarray) -> np.ndarray:
        """Apply denoising filter"""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    @staticmethod
    def preprocess(img: np.ndarray, resize: bool = True, normalize: bool = True, 
                   denoise: bool = False) -> np.ndarray:
        """Complete preprocessing pipeline"""
        processed = img.copy()
        
        if resize:
            processed = ImageProcessor.resize_image(processed)
        
        if denoise:
            processed = ImageProcessor.denoise_image(processed)
        
        if normalize:
            processed = ImageProcessor.normalize_image(processed)
        
        return processed
    
    @staticmethod
    def augment_image(img: np.ndarray) -> list:
        """Generate augmented versions of an image for better learning"""
        augmented = [img]  # Original
        
        # Slight rotations
        for angle in [-5, 5]:
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, matrix, (width, height))
            augmented.append(rotated)
        
        # Brightness variations
        for beta in [-20, 20]:
            adjusted = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
            augmented.append(adjusted)
        
        # Slight zoom
        height, width = img.shape[:2]
        zoom_factor = 1.1
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        zoomed = cv2.resize(img, (new_width, new_height))
        
        # Crop back to original size
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        cropped = zoomed[start_y:start_y + height, start_x:start_x + width]
        augmented.append(cropped)
        
        return augmented
    
    @staticmethod
    def compress_image(img: np.ndarray, quality: int = 85) -> bytes:
        """Compress image to JPEG bytes"""
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality, optimize=True)
        
        return buffer.getvalue()
    
    @staticmethod
    def extract_features_regions(img: np.ndarray) -> list:
        """Extract multiple regions for feature detection"""
        height, width = img.shape[:2]
        
        regions = [
            img,  # Full image
            img[0:height//2, :],  # Top half
            img[height//2:, :],  # Bottom half
            img[:, 0:width//2],  # Left half
            img[:, width//2:],  # Right half
            img[height//4:3*height//4, width//4:3*width//4]  # Center region
        ]
        
        return regions
