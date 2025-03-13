from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional, Literal, Any

class ScrapeRequest(BaseModel):
    """User request to scrape a website"""
    url: HttpUrl = Field(..., description="URL of the website to scrape")
    expected_data: str = Field(..., description="Description of what data the user wants to extract")
    llm_model: str = Field(..., description="LLM model to use for code generation")
    api_key: str = Field(..., description="API key for the selected LLM")

class ScrapingIssue(BaseModel):
    """Details about a scraping error"""
    error_type: str
    explanation: str
    recommendation: str

class CodeExecutionResult(BaseModel):
    """Result of code execution"""
    stdout: str
    stderr: str
    success: bool
    execution_time: float
    fix_methods: List[str] = []
    formatted_code: Optional[str] = None
    scraping_issues: Optional[List[ScrapingIssue]] = None
    scraper_type: Optional[str] = None

class ScrapeResponse(BaseModel):
    """Response with scraped data and code"""
    success: bool = Field(..., description="Whether the scraping was successful")
    data: Optional[Any] = Field(None, description="The scraped data")
    code: Optional[str] = Field(None, description="The final working code")
    attempts: int = Field(..., description="Number of attempts made")
    execution_results: List[CodeExecutionResult] = Field([], description="Results of all execution attempts")