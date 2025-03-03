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
MAX_IMAGES_PER_PAGE = 12  # Maximum images per page for multi-frame
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

def calculate_layout(num_images):
    """Calculate optimal layout based on number of images"""
    if num_images <= 1:
        return 1, 1
    elif num_images <= 4:
        return 2, 2
    else:
        return 4, 3  # Maximum grid size

def process_image_batch(images, start_idx, batch_size, pdf_canvas, layout_params):
    """Process a batch of images with memory optimization"""
    rows, cols = layout_params["rows"], layout_params["cols"]
    first_page = layout_params["first_page"]
    max_width = layout_params["max_width"]
    max_height = layout_params["max_height"]
    
    with memory_manager():
        for i in range(start_idx, min(start_idx + batch_size, len(images))):
            img = images[i]["image"]
            
            # Calculate position
            page_idx = i // MAX_IMAGES_PER_PAGE
            if page_idx > layout_params["current_page"]:
                pdf_canvas.showPage()
                layout_params["first_page"] = False
                layout_params["current_page"] = page_idx
                
            # Scale image
            img_width, img_height = img.size
            scale = min(max_width / img_width, max_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Calculate position
            pos_idx = i % MAX_IMAGES_PER_PAGE
            row = pos_idx // cols
            col = pos_idx % cols
            
            x_pos = MARGIN + col * (max_width + GRID_SPACING)
            if first_page:
                y_pos = PAGE_HEIGHT - MARGIN - (row + 1) * (max_height + GRID_SPACING) - METADATA_SPACE
            else:
                y_pos = PAGE_HEIGHT - MARGIN - (row + 1) * (max_height + GRID_SPACING)
                
            # Draw image
            try:
                pdf_canvas.drawInlineImage(img, x_pos, y_pos, width=new_width, height=new_height)
            except Exception as e:
                logger.error(f"Error drawing image {i}: {str(e)}")
                
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
        
        # Create filename
        patient_name = str(getattr(dataset, "PatientName", "Unknown")).replace("^", "_").replace(" ", "_")
        filename = filename_format.format(patient_name=patient_name, accession_number=accession_number)
        pdf_path = os.path.join(storage_dir, filename)
        
        # Create PDF canvas
        pdf_canvas = canvas.Canvas(pdf_path, pagesize=letter)
        pdf_canvas.setFont("Helvetica", 10)
        
        # Draw metadata
        y_position = draw_metadata(pdf_canvas, dataset, TOP_MARGIN)
        
        # Calculate layout
        rows, cols = calculate_layout(len(images))
        
        # Calculate maximum image dimensions
        max_width = (PAGE_WIDTH - 2 * MARGIN - (cols - 1) * GRID_SPACING) / cols
        max_height = (PAGE_HEIGHT - 2 * MARGIN - (rows - 1) * GRID_SPACING - METADATA_SPACE) / rows
        
        # Layout parameters
        layout_params = {
            "rows": rows,
            "cols": cols,
            "first_page": True,
            "current_page": 0,
            "max_width": max_width,
            "max_height": max_height
        }
        
        # Process images in batches
        for start_idx in range(0, len(images), BUFFER_SIZE):
            process_image_batch(images, start_idx, BUFFER_SIZE, pdf_canvas, layout_params)
            gc.collect()  # Force garbage collection between batches
            
        # Save the PDF
        pdf_canvas.save()
        logger.info(f"PDF saved: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        return None