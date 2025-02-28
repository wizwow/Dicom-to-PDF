# pdf_generator.py - Optimized version
import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

logger = logging.getLogger(__name__)

# Layout constants
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 50
GRID_SPACING = 10
TOP_MARGIN = 770
LINE_SPACING = 15
METADATA_SPACE = 100  # Space for metadata on first page

# Metadata fields to display
METADATA_FIELDS = [
    ("Paziente:", "PatientName", "Sconosciuto"),
    ("ID paziente:", "PatientID", "Sconosciuto"),
    ("Data di nascita:", "PatientBirthDate", "Sconosciuto"),
    ("Data studio:", "StudyDate", "Sconosciuto"),
    ("Modalità:", "Modality", "Sconosciuto"),
    ("Descrizione studio:", "StudyDescription", "Sconosciuto"),
]

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
    """
    Draw patient metadata on the PDF.
    
    Args:
        pdf_canvas: ReportLab canvas
        dataset: DICOM dataset
        y_position: Starting Y position
        
    Returns:
        int: New Y position after drawing metadata
    """
    pdf_canvas.setFont("Helvetica-Bold", 12)
    pdf_canvas.drawString(MARGIN, y_position, "Dati Paziente")
    pdf_canvas.line(MARGIN, y_position - 5, MARGIN + 200, y_position - 5)
    pdf_canvas.setFont("Helvetica", 10)
    
    y_pos = y_position - 20
    
    for label, attr, default in METADATA_FIELDS:
        value = safe_get_attribute(dataset, attr, default)
        if attr == "PatientName":
            value = value.replace("^", " ")
        pdf_canvas.drawString(MARGIN, y_pos, label)
        pdf_canvas.drawString(MARGIN + 100, y_pos, value)
        y_pos -= LINE_SPACING
        
    return y_pos

def calculate_image_position(idx, img_size, rows, cols, first_page, metadata_space):
    """
    Calculate image position in the PDF.
    
    Args:
        idx: Image index (0-based)
        img_size: (width, height) of the image
        rows, cols: Grid layout
        first_page: Whether this is the first page
        metadata_space: Space reserved for metadata
        
    Returns:
        tuple: (x_pos, y_pos) for placing the image
    """
    img_width, img_height = img_size
    images_per_page = rows * cols
    
    row = (idx % images_per_page) // cols
    col = (idx % images_per_page) % cols
    
    x_pos = MARGIN + col * (img_width + GRID_SPACING)
    
    if first_page:
        y_pos = PAGE_HEIGHT - MARGIN - (row + 1) * (img_height + GRID_SPACING) - metadata_space
    else:
        y_pos = PAGE_HEIGHT - MARGIN - (row + 1) * (img_height + GRID_SPACING)
        
    return x_pos, y_pos

def scale_image(img, max_width, max_height):
    """
    Scale an image to fit within the given dimensions.
    
    Args:
        img: PIL Image
        max_width, max_height: Maximum dimensions
        
    Returns:
        tuple: (new_width, new_height, scale_factor)
    """
    img_width, img_height = img.size
    scale = min(max_width / img_width, max_height / img_height)
    
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    return new_width, new_height, scale

def generate_pdf(storage_dir, accession_number, images, filename_format):
    """
    Generate a PDF containing DICOM images.
    
    Args:
        storage_dir: Directory to save the PDF
        accession_number: Accession number of the study
        images: List of dictionaries containing 'dataset' and 'image'
        filename_format: Format string for the output filename
        
    Returns:
        str: Path to the generated PDF or None if failed
    """
    if not images:
        logger.warning("No images provided for PDF generation")
        return None
        
    try:
        dataset = images[0]["dataset"]
        
        # Determine if it's a multi-frame DICOM
        is_multiframe = is_multiframe_dataset(dataset)
        
        # Create filename
        patient_name = safe_get_attribute(dataset, "PatientName", "Unknown").replace("^", "_").replace(" ", "_")
        filename = filename_format.format(patient_name=patient_name, accession_number=accession_number)
        pdf_path = os.path.join(storage_dir, filename)
        
        # Create PDF canvas
        pdf_canvas = canvas.Canvas(pdf_path, pagesize=letter)
        pdf_canvas.setFont("Helvetica", 10)
        
        # Determine layout based on type
        if is_multiframe:
            rows, cols = 4, 3
            images_per_page = rows * cols
            
            # Calculate maximum image size for grid
            max_width = (PAGE_WIDTH - 2 * MARGIN - (cols - 1) * GRID_SPACING) / cols
            max_height = (PAGE_HEIGHT - 2 * MARGIN - (rows - 1) * GRID_SPACING - METADATA_SPACE) / rows
        else:
            images_per_page = 1
            max_width = PAGE_WIDTH - 2 * MARGIN
            max_height = PAGE_HEIGHT - 2 * MARGIN - METADATA_SPACE
            
        # Track pagination
        first_page = True
        current_page = 0
        
        # Draw metadata on first page
        y_position = draw_metadata(pdf_canvas, dataset, TOP_MARGIN)
        
        # Process each image
        for idx, image_info in enumerate(images):
            img = image_info["image"]
            
            # Start a new page when needed
            page_idx = idx // images_per_page
            if page_idx > current_page:
                pdf_canvas.showPage()
                first_page = False
                current_page = page_idx
                
            # Scale the image
            new_width, new_height, _ = scale_image(img, max_width, max_height)
            
            if is_multiframe:
                # Grid layout
                x_pos, y_pos = calculate_image_position(
                    idx % images_per_page, 
                    (new_width, new_height), 
                    rows, cols, 
                    first_page,
                    METADATA_SPACE
                )
            else:
                # Single image centered
                x_pos = (PAGE_WIDTH - new_width) / 2
                
                if first_page:
                    y_pos = (PAGE_HEIGHT - new_height - METADATA_SPACE) / 2
                else:
                    y_pos = (PAGE_HEIGHT - new_height) / 2
                    
            # Draw the image
            try:
                pdf_canvas.drawInlineImage(img, x_pos, y_pos, width=new_width, height=new_height)
            except Exception as e:
                logger.error(f"Error drawing image {idx}: {str(e)}")
                
        # Save the PDF
        pdf_canvas.save()
        logger.info(f"PDF saved: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        return None