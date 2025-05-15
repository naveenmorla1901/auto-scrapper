from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
import os
import time
import threading
from dotenv import load_dotenv
from app.services.status_service import create_request_id, create_status, cleanup_old_statuses
from app.utils.logger import app_logger
from app.utils.dependency_checker import check_website_analyzer_dependencies
from app.services.cache_service import remove_expired_cache

# Load environment variables from .env file
load_dotenv()

# Make sure relevant env vars are loaded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found in environment variables")

from .api import endpoints
from .models.schemas import ScrapeRequest

app = FastAPI(title="Auto Web Scraper")

@app.on_event("startup")
async def startup_event():
    """Check dependencies on startup"""
    # Check if website analyzer dependencies are installed
    check_website_analyzer_dependencies()

# No middleware needed - we use a custom log filter instead

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Include API router
app.include_router(endpoints.router, prefix="/api")

# Set up periodic cleanup of old status entries
def cleanup_task():
    """Periodically clean up old status entries and expired cache"""
    while True:
        try:
            # Sleep first to avoid cleaning up immediately on startup
            time.sleep(60)  # Run every minute

            # Clean up old status entries
            cleaned_statuses = cleanup_old_statuses()
            if cleaned_statuses > 0:
                app_logger.debug(f"Cleaned up {cleaned_statuses} old status entries")
                
            # Every 10 minutes, clean up expired cache entries
            if int(time.time()) % 600 < 60:  # Run every 10 minutes
                cleaned_cache = remove_expired_cache()
                if cleaned_cache > 0:
                    app_logger.debug(f"Cleaned up {cleaned_cache} expired cache entries")
        except Exception as e:
            app_logger.error(f"Error in cleanup task: {str(e)}")

# Start the cleanup thread
cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
cleanup_thread.start()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with the scraping form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    expected_data: str = Form(...),
    llm_model: str = Form(...),
    api_key: str = Form(...),
    max_attempts: int = Form(3)  # Default to 3 attempts if not specified
):
    """
    Process the form submission and display results
    """
    try:
        # Create a request ID and initialize status tracking
        request_id = create_request_id()
        status = create_status(request_id)

        # Create scrape request
        scrape_request = ScrapeRequest(
            url=url,
            expected_data=expected_data,
            llm_model=llm_model,
            api_key=api_key
        )

        # Call the API endpoint with max_attempts parameter
        result = await endpoints.scrape_website(scrape_request, background_tasks, max_attempts=max_attempts)

        # Store validation information
        validation_info = {
            "code_success": result.code is not None,
            "data_success": result.data is not None and result.success,
        }

        # Render the results page
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "result": result,
                "url": url,
                "expected_data": expected_data,
                "attempts": max_attempts,
                "validation": validation_info,
                "request_id": request_id
            }
        )
    except HTTPException as e:
        # Handle API errors
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "status_code": e.status_code,
                "detail": e.detail
            }
        )
    except Exception as e:
        # Handle other errors
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "status_code": 500,
                "detail": str(e)
            }
        )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)