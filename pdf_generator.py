# pdf_generator.py - Optimized version
import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import gc
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Layout constants
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 50
GRID_SPACING = 10
TOP_MARGIN = 770
LINE_SPACING = 15
METADATA_SPACE = 100  # Space for metadata on first page
MAX_IMAGES_PER_PAGE_MULTIFRAME = 12  # Maximum images per page for multi-frame (4x3)
MAX_IMAGES_PER_PAGE_SINGLEFRAME = 1  # Maximum images per page for single-frame
BUFFER_SIZE = 10  # Number of images to process at once

# Metadata fields to display
METADATA_FIELDS = [
    ("Paziente:", "PatientName", "Sconosciuto"),
    ("ID paziente:", "PatientID", "Sconosciuto"),
    ("Data di nascita:", "PatientBirthDate", "Sconosciuto"),
    ("Data studio:", "StudyDate", "Sconosciuto"),
    ("Modalità:", "Modality", "Sconosciuto"),
    ("Descrizione studio:", "StudyDescription", "Sconosciuto"),
]

@contextmanager
def memory_manager():
    """Context manager for memory cleanup"""
    try:
        yield
    finally:
        gc.collect()

def is_multiframe_dataset(dataset):
    """
    Check if a DICOM dataset is a true multi-frame DICOM.
    
    Args:
        dataset: A pydicom dataset object
        
    Returns:
        bool: True if multi-frame, False otherwise
    """
    return getattr(dataset, "NumberOfFrames", 1) > 1

def safe_get_attribute(dataset, attr, default=""):
    """
    Safely get an attribute from a DICOM dataset.
    
    Args:
        dataset: A pydicom dataset object
        attr: Attribute name to retrieve
        default: Default value if attribute doesn't exist
        
    Returns:
        str: The attribute value as a string
    """
    try:
        value = getattr(dataset, attr, default)
        return str(value)
    except Exception as e:
        logger.warning(f"Error getting attribute {attr}: {str(e)}")
        return str(default)

def draw_metadata(pdf_canvas, dataset, y_position):
    """Draw metadata section with memory optimization"""
    with memory_manager():
        for label, field, default in METADATA_FIELDS:
            value = str(getattr(dataset, field, default))
            pdf_canvas.drawString(MARGIN, y_position, f"{label} {value}")
            y_position -= LINE_SPACING
    return y_position

def calculate_layout(num_images, dataset):
    """
    Calculate layout based on whether the dataset is multi-frame or single-frame.
    
    Args:
        num_images: Number of images to display
        dataset: DICOM dataset to check for multi-frame
        
    Returns:
        tuple: (rows, cols) for the layout
    """
    if is_multiframe_dataset(dataset):
        return 4, 3  # 4x3 grid for multi-frame
    else:
        return 1, 1  # 1x1 grid for single-frame

def process_image_batch(images, start_idx, batch_size, pdf_canvas, layout_params):
    """Process a batch of images with memory optimization"""
    rows, cols = layout_params["rows"], layout_params["cols"]
    first_page = layout_params["first_page"]
    max_width = layout_params["max_width"]
    max_height = layout_params["max_height"]
    max_images_per_page = layout_params["max_images_per_page"]
    
    with memory_manager():
        for i in range(start_idx, min(start_idx + batch_size, len(images))):
            img = images[i]["image"]
            
            # Calculate position
            page_idx = i // max_images_per_page
            if page_idx > layout_params["current_page"]:
                logger.info(f"Creating new page {page_idx + 1}")
                pdf_canvas.showPage()
                layout_params["first_page"] = False
                layout_params["current_page"] = page_idx
                
            # Scale image
            img_width, img_height = img.size
            scale = min(max_width / img_width, max_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            logger.debug(f"Image {i + 1}: Original size {img_width}x{img_height}, Scaled to {new_width}x{new_height}")
            
            # Calculate position
            if rows == 1 and cols == 1:  # Single frame case
                # Center the image on the page
                x_pos = (PAGE_WIDTH - new_width) / 2
                y_pos = (PAGE_HEIGHT - new_height) / 2
                if first_page:
                    y_pos -= METADATA_SPACE / 2  # Adjust for metadata space on first page
                logger.debug(f"Single frame image {i + 1}: Centered at position ({x_pos}, {y_pos})")
            else:  # Multi-frame case
                # Calculate position in grid for current page
                current_page_pos = i % max_images_per_page
                row = current_page_pos // cols
                col = current_page_pos % cols
                
                x_pos = MARGIN + col * (max_width + GRID_SPACING)
                if first_page and page_idx == 0:
                    y_pos = PAGE_HEIGHT - MARGIN - (row + 1) * (max_height + GRID_SPACING) - METADATA_SPACE
                else:
                    y_pos = PAGE_HEIGHT - MARGIN - (row + 1) * (max_height + GRID_SPACING)
                logger.debug(f"Multi-frame image {i + 1}: Grid position ({row}, {col}) at ({x_pos}, {y_pos})")
                
            # Draw image
            try:
                pdf_canvas.drawInlineImage(img, x_pos, y_pos, width=new_width, height=new_height)
                logger.debug(f"Successfully drew image {i + 1} on page {page_idx + 1}")
            except Exception as e:
                logger.error(f"Error drawing image {i + 1} on page {page_idx + 1}: {str(e)}")
                
            # Clear image from memory
            del img
            gc.collect()

def generate_pdf(storage_dir, accession_number, images, filename_format):
    """Generate a PDF with memory optimization"""
    if not images:
        logger.warning("No images provided for PDF generation")
        return None
        
    try:
        dataset = images[0]["dataset"]
        logger.info(f"Starting PDF generation for patient: {getattr(dataset, 'PatientName', 'Unknown')}")
        logger.info(f"Total images to process: {len(images)}")
        
        # Create filename
        patient_name = str(getattr(dataset, "PatientName", "Unknown")).replace("^", "_").replace(" ", "_")
        filename = filename_format.format(patient_name=patient_name, accession_number=accession_number)
        pdf_path = os.path.join(storage_dir, filename)
        logger.info(f"PDF will be saved as: {pdf_path}")
        
        # Create PDF canvas
        pdf_canvas = canvas.Canvas(pdf_path, pagesize=letter)
        pdf_canvas.setFont("Helvetica", 10)
        
        # Draw metadata
        y_position = draw_metadata(pdf_canvas, dataset, TOP_MARGIN)
        logger.debug("Metadata section drawn successfully")
        
        # Calculate layout
        rows, cols = calculate_layout(len(images), dataset)
        is_multiframe = is_multiframe_dataset(dataset)
        logger.info(f"Layout configuration: {'Multi-frame' if is_multiframe else 'Single-frame'} mode")
        logger.info(f"Grid layout: {rows}x{cols}")
        
        # Set max images per page based on whether it's multi-frame
        max_images_per_page = MAX_IMAGES_PER_PAGE_MULTIFRAME if is_multiframe else MAX_IMAGES_PER_PAGE_SINGLEFRAME
        logger.info(f"Maximum images per page: {max_images_per_page}")
        
        # Calculate maximum image dimensions
        max_width = (PAGE_WIDTH - 2 * MARGIN - (cols - 1) * GRID_SPACING) / cols
        max_height = (PAGE_HEIGHT - 2 * MARGIN - (rows - 1) * GRID_SPACING - METADATA_SPACE) / rows
        logger.debug(f"Maximum image dimensions: {max_width}x{max_height}")
        
        # Layout parameters
        layout_params = {
            "rows": rows,
            "cols": cols,
            "first_page": True,
            "current_page": 0,
            "max_width": max_width,
            "max_height": max_height,
            "max_images_per_page": max_images_per_page
        }
        
        # Process images in batches
        total_batches = (len(images) + BUFFER_SIZE - 1) // BUFFER_SIZE
        logger.info(f"Processing {len(images)} images in {total_batches} batches")
        
        for batch_idx, start_idx in enumerate(range(0, len(images), BUFFER_SIZE), 1):
            logger.info(f"Processing batch {batch_idx}/{total_batches} (images {start_idx + 1}-{min(start_idx + BUFFER_SIZE, len(images))})")
            process_image_batch(images, start_idx, BUFFER_SIZE, pdf_canvas, layout_params)
            gc.collect()  # Force garbage collection between batches
            
        # Save the PDF
        pdf_canvas.save()
        logger.info(f"PDF successfully generated and saved: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        return None