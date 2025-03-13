from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Make sure relevant env vars are loaded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found in environment variables")

from .api import endpoints
from .models.schemas import ScrapeRequest

app = FastAPI(title="Auto Web Scraper")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Include API router
app.include_router(endpoints.router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with the scraping form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request: Request,
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
        # Create scrape request
        scrape_request = ScrapeRequest(
            url=url,
            expected_data=expected_data,
            llm_model=llm_model,
            api_key=api_key
        )
        
        # Call the API endpoint with max_attempts parameter
        result = await endpoints.scrape_website(scrape_request, max_attempts=max_attempts)
        
        # Render the results page
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "result": result,
                "url": url,
                "expected_data": expected_data,
                "attempts": max_attempts
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