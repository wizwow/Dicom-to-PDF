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

def calculate_layout(dataset):
    """
    Calculate layout based on whether the dataset is multi-frame or single-frame.
    
    Args:
        dataset: DICOM dataset to check for multi-frame
        
    Returns:
        tuple: (rows, cols) for the layout
    """
    if is_multiframe_dataset(dataset):
        return 4, 3  # 4x3 grid for multi-frame
    else:
        return 1, 1  # 1x1 grid for single-frame

def draw_page_number(pdf_canvas, page_num):
    """Draw page number at the bottom center of the page"""
    text = f"Pagina {page_num}"
    text_width = pdf_canvas.stringWidth(text, "Helvetica", 10)
    x = (PAGE_WIDTH - text_width) / 2
    pdf_canvas.drawString(x, MARGIN / 2, text)

def process_image_batch(images, start_idx, batch_size, pdf_canvas, layout_params):
    """Process a batch of images with memory optimization"""
    max_width = layout_params["max_width"]
    max_height = layout_params["max_height"]
    first_page = layout_params["first_page"]
    current_page = layout_params["current_page"]
    last_image_type = layout_params.get("last_image_type", None)
    
    with memory_manager():
        for i in range(start_idx, min(start_idx + batch_size, len(images))):
            img = images[i]["image"]
            dataset = images[i]["dataset"]
            
            # Calculate layout for this specific image
            rows, cols = calculate_layout(dataset)
            is_multiframe = is_multiframe_dataset(dataset)
            current_image_type = "multi" if is_multiframe else "single"
            
            # Force new page if switching between single and multi-frame
            if last_image_type is not None and current_image_type != last_image_type:
                draw_page_number(pdf_canvas, current_page + 1)
                pdf_canvas.showPage()
                layout_params["first_page"] = False
                current_page += 1
                layout_params["current_page"] = current_page
                layout_params["multi_frame_count"] = 0
            
            # Update last image type
            layout_params["last_image_type"] = current_image_type
            
            # Calculate position
            page_idx = current_page
            if current_image_type == "single":  # Single frame case
                # Each single frame image gets its own page
                if i > start_idx or not first_page:
                    draw_page_number(pdf_canvas, current_page + 1)
                    pdf_canvas.showPage()
                    layout_params["first_page"] = False
                    current_page += 1
                    layout_params["current_page"] = current_page
                
                # Use full page dimensions for single frame
                available_width = PAGE_WIDTH - 2 * MARGIN
                available_height = PAGE_HEIGHT - 2 * MARGIN - (METADATA_SPACE if first_page and page_idx == 0 else 0)
                
                # Scale image
                img_width, img_height = img.size
                scale = min(available_width / img_width, available_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # Center the image on the page
                x_pos = (PAGE_WIDTH - new_width) / 2
                y_pos = (PAGE_HEIGHT - new_height) / 2
                if first_page and page_idx == 0:
                    y_pos -= METADATA_SPACE / 2
                    
            else:  # Multi-frame case
                # Check if we need a new page
                pos_on_current_page = layout_params.get("multi_frame_count", 0)
                if pos_on_current_page >= MAX_IMAGES_PER_PAGE_MULTIFRAME:
                    draw_page_number(pdf_canvas, current_page + 1)
                    pdf_canvas.showPage()
                    layout_params["first_page"] = False
                    current_page += 1
                    layout_params["current_page"] = current_page
                    pos_on_current_page = 0
                    layout_params["multi_frame_count"] = 0
                
                # Calculate available space for each image in the grid
                available_width = (PAGE_WIDTH - 2 * MARGIN - (cols - 1) * GRID_SPACING) / cols
                if first_page and page_idx == 0:
                    available_height = (PAGE_HEIGHT - 2 * MARGIN - (rows - 1) * GRID_SPACING - METADATA_SPACE) / rows
                else:
                    available_height = (PAGE_HEIGHT - 2 * MARGIN - (rows - 1) * GRID_SPACING) / rows
                
                # Scale image
                img_width, img_height = img.size
                scale = min(available_width / img_width, available_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # Calculate position in grid
                row = pos_on_current_page // cols
                col = pos_on_current_page % cols
                
                x_pos = MARGIN + col * (available_width + GRID_SPACING)
                if first_page and page_idx == 0:
                    y_pos = PAGE_HEIGHT - MARGIN - (row + 1) * (available_height + GRID_SPACING) - METADATA_SPACE
                else:
                    y_pos = PAGE_HEIGHT - MARGIN - (row + 1) * (available_height + GRID_SPACING)
                
                # Update multi-frame counter
                layout_params["multi_frame_count"] = pos_on_current_page + 1
            
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
        
        # Layout parameters - these will be recalculated for each image based on its type
        layout_params = {
            "first_page": True,
            "current_page": 0,
            "max_width": PAGE_WIDTH - 2 * MARGIN,  # Will be adjusted for grid layout when needed
            "max_height": PAGE_HEIGHT - 2 * MARGIN - METADATA_SPACE,  # Will be adjusted for grid layout when needed
            "multi_frame_count": 0,  # Counter for multi-frame images on current page
            "last_image_type": None  # Track the type of the last image processed
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