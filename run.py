"""
Run script for the Auto Web Scraper application.
"""
import uvicorn
import logging

# Configure a simple filter for status endpoint logs
class StatusEndpointFilter(logging.Filter):
    def filter(self, record):
        # Skip logging for status endpoint requests
        try:
            if hasattr(record, 'args') and len(record.args) >= 3:
                request_path = str(record.args[1])  # The URL path is usually the second argument
                if '/api/status/' in request_path:
                    return False
        except Exception:
            pass  # If there's any error in filtering, just allow the log
        return True

# Configure logging
def configure_logging():
    # Configure the root logger to avoid duplicate logs
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Get the uvicorn access logger
    access_logger = logging.getLogger("uvicorn.access")

    # Add our custom filter
    access_logger.addFilter(StatusEndpointFilter())

    return access_logger

if __name__ == "__main__":
    # Configure logging
    access_logger = configure_logging()

    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True,
        log_level="info"
    )
