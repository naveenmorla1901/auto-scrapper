"""
Custom logging configuration for the application.
"""
import logging

# Create a simple filter that skips status endpoint logs
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

# Configure the uvicorn logging
def configure_uvicorn_logging():
    """Configure uvicorn logging to filter status endpoint requests."""
    # Get the uvicorn access logger
    access_logger = logging.getLogger("uvicorn.access")

    # Add our custom filter
    status_filter = StatusEndpointFilter()
    access_logger.addFilter(status_filter)

    return access_logger
