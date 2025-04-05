from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional, Literal, Any, Union

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

class ProcessStage(BaseModel):
    """Information about a processing stage"""
    name: str = Field(..., description="Name of the processing stage")
    start_time: float = Field(..., description="Start time of the stage (timestamp)")
    end_time: Optional[float] = Field(None, description="End time of the stage (timestamp)")
    duration: Optional[float] = Field(None, description="Duration of the stage in seconds")
    status: str = Field("in_progress", description="Status of the stage (in_progress, completed, failed)")
    details: Optional[str] = Field(None, description="Additional details about the stage")

class StatusResponse(BaseModel):
    """Response with current process status"""
    request_id: str = Field(..., description="Unique ID for the request")
    status: str = Field(..., description="Overall status of the process")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    progress: float = Field(0.0, description="Progress percentage (0-100)")
    stages: List[ProcessStage] = Field([], description="Information about each processing stage")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message if any")

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

class TokenUsage(BaseModel):
    """Token usage information for an LLM call"""
    input_tokens: int = Field(0, description="Number of input tokens used")
    output_tokens: int = Field(0, description="Number of output tokens used")
    total_tokens: int = Field(0, description="Total number of tokens used")
    cost: float = Field(0.0, description="Estimated cost in USD")

class ScrapeResponse(BaseModel):
    """Response with scraped data and code"""
    success: bool = Field(..., description="Whether the scraping was successful")
    data: Optional[Any] = Field(None, description="The scraped data")
    code: Optional[str] = Field(None, description="The final working code")
    attempts: int = Field(..., description="Number of attempts made")
    execution_results: List[CodeExecutionResult] = Field([], description="Results of all execution attempts")
    helper_llm_usage: Optional[TokenUsage] = Field(None, description="Token usage for the helper LLM")
    coding_llm_usage: Optional[TokenUsage] = Field(None, description="Token usage for the coding LLM")
    total_cost: Optional[float] = Field(None, description="Total estimated cost in USD")
    process_stages: List[ProcessStage] = Field([], description="Information about each processing stage")