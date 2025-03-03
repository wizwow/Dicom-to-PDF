# receiver.py - Optimized version
import os
import threading
import time
import logging
from collections import defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import psutil
from pynetdicom import AE, evt, AllStoragePresentationContexts
from pynetdicom.sop_class import Verification
from pdf_generator import generate_pdf
from image_extractor import extract_images

logger = logging.getLogger(__name__)

def measure_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"Performance - {func.__name__}: "
                   f"Time: {end_time - start_time:.2f}s, "
                   f"Memory: {end_memory - start_memory:.2f}MB")
        return result
    return wrapper

class DICOMToPDFReceiver:
    def __init__(self, storage_dir, timeout=60, filename_format="{patient_name}_{accession_number}.pdf", ae_title="DICOMHLT", max_workers=4):
        self.storage_dir = storage_dir
        self.timeout = timeout
        self.filename_format = filename_format
        self.ae_title = ae_title
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Thread-safe data structures
        self.lock = Lock()
        self.image_buffer = defaultdict(list)
        self.last_received_time = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_futures = {}
        
        # Control flag for the cleanup thread
        self.running = True
        self.cleanup_thread = threading.Thread(target=self.check_timeouts, daemon=True)
        self.cleanup_thread.start()
        
        self.ae = None  # Will be set in start_server

    @measure_performance
    def handle_store(self, event):
        """
        Handle incoming DICOM C-STORE requests by extracting images
        and storing them in the image buffer.
        """
        try:
            dataset = event.dataset
            dataset.file_meta = event.file_meta
            
            # Extract key identifier
            accession_number = getattr(dataset, "AccessionNumber", "NoAccession")
            
            # Submit image extraction to thread pool
            future = self.executor.submit(self._process_dataset, dataset, accession_number)
            self.processing_futures[accession_number] = future
            
            # Success status
            return 0x0000
            
        except Exception as e:
            logger.error(f"Error in handle_store: {str(e)}", exc_info=True)
            # Error status - Processing failure
            return 0xC210

    def _process_dataset(self, dataset, accession_number):
        """
        Process a DICOM dataset in a separate thread.
        """
        try:
            # Extract images from the dataset
            images = extract_images(dataset)
            
            if images:
                with self.lock:
                    for img in images:
                        self.image_buffer[accession_number].append({"dataset": dataset, "image": img})
                    self.last_received_time[accession_number] = time.time()
                    logger.info(f"Received images for accession {accession_number}, "
                              f"total: {len(self.image_buffer[accession_number])}")
            else:
                logger.warning(f"No images extracted from dataset for accession {accession_number}")
                
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}", exc_info=True)

    def check_timeouts(self):
        """
        Periodically check for studies that have timed out and need to be finalized.
        """
        while self.running:
            try:
                to_finalize = []
                current_time = time.time()
                
                # Get list of accessions to finalize under thread safety
                with self.lock:
                    to_finalize = [
                        accession for accession, last_time in self.last_received_time.items()
                        if current_time - last_time > self.timeout
                    ]
                
                # Process finalizations
                for accession in to_finalize:
                    try:
                        # Get data under lock
                        with self.lock:
                            images = self.image_buffer.get(accession, []).copy()
                        
                        # Generate PDF outside of lock to reduce contention
                        if images:
                            self.finalize_pdf(accession, images)
                        
                        # Remove data under lock
                        with self.lock:
                            if accession in self.image_buffer:
                                del self.image_buffer[accession]
                            if accession in self.last_received_time:
                                del self.last_received_time[accession]
                                
                    except Exception as e:
                        logger.error(f"Error finalizing PDF for accession {accession}: {str(e)}", exc_info=True)
                        
            except Exception as e:
                logger.error(f"Error in check_timeouts: {str(e)}", exc_info=True)
                
            time.sleep(1)
            
    @measure_performance
    def finalize_pdf(self, accession_number, images=None):
        """
        Generate a PDF for a completed study.
        """
        try:
            # If images not provided, get from buffer
            if images is None:
                with self.lock:
                    images = self.image_buffer.get(accession_number, []).copy()
                    
            if not images:
                logger.warning(f"No images to finalize for accession {accession_number}")
                return
                
            logger.info(f"Finalizing PDF for accession {accession_number} with {len(images)} images")
            
            # Submit PDF generation to thread pool
            future = self.executor.submit(generate_pdf, 
                                       self.storage_dir, 
                                       accession_number, 
                                       images, 
                                       self.filename_format)
            
            # Wait for PDF generation to complete
            pdf_path = future.result()
            
            if pdf_path:
                logger.info(f"PDF generation completed: {pdf_path}")
            else:
                logger.error(f"PDF generation failed for accession {accession_number}")
                
        except Exception as e:
            logger.error(f"Error in finalize_pdf: {str(e)}", exc_info=True)
            
    def start_server(self, port=11112, address='0.0.0.0'):

        try:
            self.ae = AE(ae_title=self.ae_title)
            
            # Add presentation contexts - only need to do this once
            for context in AllStoragePresentationContexts:
                self.ae.add_supported_context(context.abstract_syntax, [
                    '1.2.840.10008.1.2',    # Implicit VR Little Endian
                    '1.2.840.10008.1.2.1',  # Explicit VR Little Endian
                    '1.2.840.10008.1.2.2',  # Explicit VR Big Endian
                    '1.2.840.10008.1.2.4.50',  # JPEG Baseline (Process 1)
                    '1.2.840.10008.1.2.4.51',  # JPEG Extended (Process 2 & 4)
                    '1.2.840.10008.1.2.4.70',  # JPEG Lossless (Non-Hierarchical, First-Order Prediction)

                ])
                
            # Add verification service
            self.ae.add_supported_context(Verification)
            
            # Set up event handlers
            handlers = [(evt.EVT_C_STORE, self.handle_store)]
            
            logger.info(f"🏥 DICOM C-STORE SCP Server attivo sulla porta {port} con AE title {self.ae_title}")
            
            # Start server
            self.server = self.ae.start_server((address, port), block=True, evt_handlers=handlers)
            
        except Exception as e:
            logger.error(f"Avvio DICOM server fallito: {str(e)}", exc_info=True)
            raise
            
    def shutdown(self):
        """
        Gracefully shut down the server.
        """
        logger.info("Shutting down DICOM receiver...")
        
        # Stop the cleanup thread
        self.running = False
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
            
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
            
        # Finalize any remaining studies
        with self.lock:
            accessions = list(self.image_buffer.keys())
            
        for accession in accessions:
            try:
                self.finalize_pdf(accession)
            except Exception as e:
                logger.error(f"Error finalizing PDF during shutdown: {str(e)}")
                
        # Stop the server if it's running
        if hasattr(self, 'server') and self.server:
            self.server.shutdown()
            self.server = None  # Clear server reference