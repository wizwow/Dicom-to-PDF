# image_extractor.py - Optimized version
import logging
import numpy as np
import cv2
from PIL import Image

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

def extract_images(dataset):
    """
    Extract images from a DICOM dataset.
    
    Args:
        dataset: A pydicom dataset object
        
    Returns:
        List of PIL Image objects
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
            
        # Get pixel array
        try:
            pixel_array = dataset.pixel_array
        except Exception as e:
            logger.error(f"Error accessing pixel array: {str(e)}")
            return images
            
        # Handle different photometric interpretations
        try:
            # Convert to float32 for processing
            pixel_array = np.array(pixel_array, dtype=np.float32)
            
            # Special handling for PALETTE COLOR
            if photometric == "PALETTE COLOR":
                try:
                    dataset.convert_pixel_data()
                    pixel_array = dataset.pixel_array
                    img = Image.fromarray(pixel_array, mode="RGB")
                    images.append(img)
                    return images
                except Exception as e:
                    logger.error(f"Error converting PALETTE COLOR: {str(e)}")
                    # Fall back to grayscale
                    photometric = "MONOCHROME2"
                    
            # Handle MONOCHROME1 (inverted grayscale)
            if photometric == "MONOCHROME1":
                if np.isnan(pixel_array).any():
                    logger.warning("NaN values found in pixel array")
                    pixel_array = np.nan_to_num(pixel_array)
                    
                if pixel_array.size > 0:
                    pixel_max = np.max(pixel_array)
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
                pixel_min = np.min(pixel_array)
                pixel_max = np.max(pixel_array)
                
                if pixel_max > 255 or pixel_min < 0:
                    # Avoid division by zero
                    if pixel_max > pixel_min:
                        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min)) * 255
                    else:
                        pixel_array = np.zeros_like(pixel_array)
                        
            # Convert to uint8 for PIL
            pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
            
            # Handle multi-frame images
            num_frames = getattr(dataset, "NumberOfFrames", 1)
            
            if num_frames > 1:
                for i in range(num_frames):
                    try:
                        frame = pixel_array[i]
                        
                        # Convert YBR to RGB if needed
                        if photometric in ["YBR_FULL", "YBR_FULL_422", "YBR_PARTIAL_422"]:
                            frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2RGB)
                            
                        # Create PIL image
                        mode = "RGB" if photometric in ["RGB", "YBR_FULL", "YBR_FULL_422", "YBR_PARTIAL_422"] else "L"
                        img = Image.fromarray(frame, mode=mode)
                        images.append(img)
                    except Exception as e:
                        logger.error(f"Error processing frame {i}: {str(e)}")
            else:
                # Single frame
                try:
                    # Convert YBR to RGB if needed
                    if photometric in ["YBR_FULL", "YBR_FULL_422", "YBR_PARTIAL_422"]:
                        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_YCrCb2RGB)
                        
                    # Create PIL image
                    mode = "RGB" if photometric in ["RGB", "YBR_FULL", "YBR_FULL_422", "YBR_PARTIAL_422"] else "L"
                    img = Image.fromarray(pixel_array, mode=mode)
                    images.append(img)
                except Exception as e:
                    logger.error(f"Error creating single frame image: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing pixel data: {str(e)}", exc_info=True)
            
    except Exception as e:
        logger.error(f"Unexpected error in extract_images: {str(e)}", exc_info=True)
        
    return images