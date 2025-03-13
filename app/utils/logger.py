"""
Centralized logging setup for the application
"""
import logging
import sys
import os
import platform
from datetime import datetime

# Check if running on Windows
IS_WINDOWS = platform.system() == "Windows"

class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors to the log messages based on level"""
    
    # ANSI color codes
    COLOR_BLUE = '\033[94m'
    COLOR_GREEN = '\033[92m'
    COLOR_YELLOW = '\033[93m'
    COLOR_RED = '\033[91m'
    COLOR_MAGENTA = '\033[95m'
    COLOR_BOLD = '\033[1m'
    COLOR_RESET = '\033[0m'
    
    FORMATS = {
        logging.DEBUG: COLOR_BLUE + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + COLOR_RESET,
        logging.INFO: COLOR_GREEN + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + COLOR_RESET,
        logging.WARNING: COLOR_YELLOW + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + COLOR_RESET,
        logging.ERROR: COLOR_RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + COLOR_RESET,
        logging.CRITICAL: COLOR_BOLD + COLOR_MAGENTA + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + COLOR_RESET
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name, log_to_file=True):
    """
    Setup a logger with console and file handlers
    
    Args:
        name: Logger name (usually __name__)
        log_to_file: Whether to save logs to file in addition to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # On Windows, we don't use the ColorFormatter due to encoding issues
    if IS_WINDOWS:
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    else:
        console_handler.setFormatter(ColorFormatter())
        
    logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_to_file:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger

# Main application logger
app_logger = setup_logger('auto_scraper')

def log_llm_interaction(direction, model, prompt=None, response=None):
    """
    Special logger for LLM interactions with detailed formatting
    
    Args:
        direction: "INPUT" or "OUTPUT"
        model: The model being used
        prompt: The prompt sent to the model (for INPUT)
        response: The response from the model (for OUTPUT)
    """
    # Use different markers based on platform to avoid encoding issues on Windows
    if IS_WINDOWS:
        input_marker = ">> LLM REQUEST TO"
        output_marker = "<< LLM RESPONSE FROM"
    else:
        input_marker = "🔹 LLM REQUEST TO"
        output_marker = "🔸 LLM RESPONSE FROM"
    
    if direction.upper() == "INPUT":
        app_logger.info(f"{input_marker} {model}")
        app_logger.info(f"PROMPT:\n{'-'*80}\n{prompt}\n{'-'*80}")
    else:  # OUTPUT
        app_logger.info(f"{output_marker} {model}")
        app_logger.info(f"RESPONSE:\n{'-'*80}\n{response[:500]}{'...' if len(response) > 500 else ''}\n{'-'*80}")

def log_code_execution(code, result):
    """
    Special logger for code execution with detailed formatting
    
    Args:
        code: The code being executed
        result: The result of the execution
    """
    # Use different markers based on platform to avoid encoding issues on Windows
    if IS_WINDOWS:
        exec_marker = "EXECUTING CODE"
        success_marker = "CODE EXECUTION SUCCESSFUL"
        fail_marker = "CODE EXECUTION FAILED"
    else:
        exec_marker = "⚙️ EXECUTING CODE ⚙️"
        success_marker = "✅ CODE EXECUTION SUCCESSFUL ✅"
        fail_marker = "❌ CODE EXECUTION FAILED ❌"
    
    app_logger.info(f"{exec_marker}")
    app_logger.info(f"CODE:\n{'-'*80}\n{code[:500]}{'...' if len(code) > 500 else ''}\n{'-'*80}")
    
    success = result.get('success', False)
    stdout = result.get('stdout', '')
    stderr = result.get('stderr', '')
    
    if success:
        app_logger.info(f"{success_marker}")
    else:
        app_logger.error(f"{fail_marker}")
    
    if stdout:
        app_logger.info(f"STDOUT:\n{'-'*80}\n{stdout[:500]}{'...' if len(stdout) > 500 else ''}\n{'-'*80}")
    
    if stderr:
        app_logger.error(f"STDERR:\n{'-'*80}\n{stderr[:500]}{'...' if len(stderr) > 500 else ''}\n{'-'*80}")