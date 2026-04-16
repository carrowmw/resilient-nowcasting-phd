import logging
import sys
import os
import traceback
from datetime import datetime

def setup_logger(experiment_name: str) -> logging.Logger:
    """
    Sets up a logger that outputs to both the console and a file in the logs/ directory.
    Crucially, overrides sys.excepthook to log unhandled exceptions and failed assertions
    before the script crashes.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logger = logging.getLogger() # Root logger
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Writing to {log_file}")
    
    # CRITICAL: Override sys.excepthook for offensive programming crash-logging
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow Ctrl+C to work normally without logging as an error
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical("Uncaught exception or assertion failure!", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Also print to standard error for immediate terminal visibility
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

    sys.excepthook = handle_exception
    
    return logger
