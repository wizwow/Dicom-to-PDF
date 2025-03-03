# image_extractor.py - Optimized version
import logging
import numpy as np
import cv2
from PIL import Image
import gc
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Define supported photometric interpretations and their handling methods
SUPPORTED_PHOTOMETRIC = {
    "MONOCHROME1": "monochrome1",
    "MONOCHROME2": "monochrome2",
    "RGB": "rgb",
    "PALETTE COLOR": "palette_color",
    "YBR_FULL": "ybr_full",
    "YBR_FULL_422": "ybr_full",  # Treat as YBR_FULL
    "YBR_PARTIAL_422": "ybr_full"  # Treat as YBR_FULL
}

# Constants for chunk processing
CHUNK_SIZE = 5  # Process 5 frames at a time for multi-frame images
MAX_IMAGE_DIM = 4096  # Maximum dimension for image processing

@contextmanager
def memory_manager():
    """Context manager for memory cleanup"""
    try:
        yield
    finally:
        gc.collect()

def downsample_if_needed(image, max_dim=MAX_IMAGE_DIM):
    """
    Downsample image if dimensions exceed max_dim
    """
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        if height > max_dim or width > max_dim:
            scale = min(max_dim / height, max_dim / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def process_pixel_array(pixel_array, photometric, dataset):
    """
    Process pixel array with memory optimization
    """
    with memory_manager():
        # Convert to float32 for processing
        pixel_array = np.array(pixel_array, dtype=np.float32)
        
        # Handle MONOCHROME1 (inverted grayscale)
        if photometric == "MONOCHROME1":
            if np.isnan(pixel_array).any():
                pixel_array = np.nan_to_num(pixel_array)
                
            if pixel_array.size > 0:
                pixel_max = float(np.max(pixel_array))
                pixel_array = pixel_max - pixel_array
                
        # Apply rescale slope and intercept if available
        try:
            if hasattr(dataset, 'RescaleSlope') and hasattr(dataset, 'RescaleIntercept'):
                slope = float(dataset.RescaleSlope)
                intercept = float(dataset.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
        except Exception as e:
            logger.warning(f"Error applying rescale parameters: {str(e)}")
            
        # Normalize to 8-bit range if needed
        if pixel_array.size > 0:
            pixel_min = float(np.min(pixel_array))
            pixel_max = float(np.max(pixel_array))
            
            if pixel_max > pixel_min:
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
            else:
                pixel_array = np.zeros_like(pixel_array)
                
        # Convert to uint8
        pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
        
        return pixel_array

def process_frame(frame, photometric):
    """
    Process a single frame
    """
    with memory_manager():
        # Downsample if needed
        frame = downsample_if_needed(frame)
        
        # Convert YBR to RGB if needed
        if photometric in ["YBR_FULL", "YBR_FULL_422", "YBR_PARTIAL_422"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2RGB)
            
        # Create PIL image
        mode = "RGB" if photometric in ["RGB", "YBR_FULL", "YBR_FULL_422", "YBR_PARTIAL_422"] else "L"
        return Image.fromarray(frame, mode=mode)

def extract_images(dataset):
    """
    Extract images from a DICOM dataset with memory optimization
    """
    images = []
    
    try:
        # Check if dataset has pixel data
        if not hasattr(dataset, 'PixelData'):
            logger.warning("Dataset has no pixel data")
            return images
            
        # Get photometric interpretation
        photometric = getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")
        
        # Check if photometric interpretation is supported
        if photometric not in SUPPORTED_PHOTOMETRIC:
            logger.warning(f"Unsupported photometric interpretation: {photometric}")
            photometric = "MONOCHROME2"  # Default to MONOCHROME2
            
        # Special handling for PALETTE COLOR
        if photometric == "PALETTE COLOR":
            try:
                with memory_manager():
                    dataset.convert_pixel_data()
                    pixel_array = dataset.pixel_array
                    img = Image.fromarray(pixel_array, mode="RGB")
                    images.append(img)
                    return images
            except Exception as e:
                logger.error(f"Error converting PALETTE COLOR: {str(e)}")
                photometric = "MONOCHROME2"
                
        # Get pixel array
        try:
            pixel_array = dataset.pixel_array
        except Exception as e:
            logger.error(f"Error accessing pixel array: {str(e)}")
            return images
            
        # Process pixel array
        pixel_array = process_pixel_array(pixel_array, photometric, dataset)
        
        # Handle multi-frame images
        num_frames = getattr(dataset, "NumberOfFrames", 1)
        
        if num_frames > 1:
            # Process frames in chunks
            for chunk_start in range(0, num_frames, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, num_frames)
                
                with memory_manager():
                    for i in range(chunk_start, chunk_end):
                        try:
                            frame = pixel_array[i]
                            img = process_frame(frame, photometric)
                            images.append(img)
                        except Exception as e:
                            logger.error(f"Error processing frame {i}: {str(e)}")
                            continue
                            
                # Clear memory after each chunk
                gc.collect()
        else:
            # Single frame
            try:
                img = process_frame(pixel_array, photometric)
                images.append(img)
            except Exception as e:
                logger.error(f"Error creating single frame image: {str(e)}")
                
    except Exception as e:
        logger.error(f"Unexpected error in extract_images: {str(e)}", exc_info=True)
        
    return images