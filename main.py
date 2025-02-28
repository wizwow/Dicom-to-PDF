# main.py - Optimized version
import argparse
import logging
import signal
import sys
from receiver import DICOMToPDFReceiver

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def parse_args():
    parser = argparse.ArgumentParser(description='DICOM to PDF conversion server')
    parser.add_argument('--storage-dir', type=str, default='dicom_pdfs',
                        help='Directory to store generated PDFs')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout in seconds before finalizing a study')
    parser.add_argument('--port', type=int, default=11112,
                        help='Port to listen on')
    parser.add_argument('--ae-title', type=str, default='DICOMHLT',
                        help='AE Title for the DICOM server')
    return parser.parse_args()

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    
    receiver = DICOMToPDFReceiver(
        storage_dir=args.storage_dir, 
        timeout=args.timeout,
        ae_title=args.ae_title
    )
    
    # Setup graceful shutdown
    def signal_handler(sig, frame):
        logging.info("Shutting down server...")
        receiver.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logging.info(f"Starting DICOM to PDF conversion server on port {args.port}")
    receiver.start_server(port=args.port)