from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Optional, Literal, Any, Union, Set
from collections import defaultdict

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

class SelectorStrategy(BaseModel):
    """Selector strategy with reliability score"""
    selector: str = Field(..., description="CSS selector")
    reliability: float = Field(..., description="Reliability score (0-1)")
    method: str = Field(..., description="Method used to generate the selector")
    count: Optional[int] = Field(None, description="Number of elements matching this selector")
    note: Optional[str] = Field(None, description="Additional notes about the selector")

class ElementSelectorStrategies(BaseModel):
    """Selector strategies for an element"""
    element: str = Field(..., description="Element type")
    strategies: List[SelectorStrategy] = Field(..., description="List of selector strategies")

class ContentPattern(BaseModel):
    """Content pattern detected in the website"""
    parent_selector: str = Field(..., description="Selector for the parent element")
    child_selector: str = Field(..., description="Selector for the child elements")
    count: int = Field(..., description="Number of child elements")
    confidence: str = Field(..., description="Confidence level (high, medium, low)")

class ExtractionTestResult(BaseModel):
    """Result of testing extraction patterns"""
    pattern: str = Field(..., description="Pattern name")
    selector: str = Field(..., description="Selector used")
    sample: List[str] = Field(..., description="Sample of extracted content")
    count: int = Field(..., description="Number of elements found")

class InteractiveElement(BaseModel):
    """Interactive element detected on the page"""
    selector: str = Field(..., description="Selector for the element")
    text: str = Field(..., description="Text content of the element")
    type: str = Field(..., description="Type of interactive element")

class SectionScreenshot(BaseModel):
    """Screenshot of a section of the page"""
    selector: str = Field(..., description="Selector for the section")
    path: str = Field(..., description="Path to the screenshot file")
    dimensions: str = Field(..., description="Dimensions of the section")

class VisualStructure(BaseModel):
    """Visual structure analysis of the page"""
    full_screenshot: Optional[str] = Field(None, description="Path to the full page screenshot")
    sections: List[SectionScreenshot] = Field(default_factory=list, description="Screenshots of page sections")

class CodeTemplate(BaseModel):
    """Code template information"""
    selectors: Dict[str, str] = Field(default_factory=dict, description="Selectors to use in templates")
    template_type: Optional[str] = Field(None, description="Type of template to use")

class WebsiteAnalysis(BaseModel):
    """Enhanced website analysis information"""
    url: str = Field(..., description="URL of the analyzed website")
    is_dynamic: bool = Field(False, description="Whether the website uses dynamic content")
    
    # Structure Analysis
    content_hierarchy: Dict[str, int] = Field(default_factory=dict, description="Content hierarchy information")
    components: Dict[str, int] = Field(default_factory=dict, description="Component counts")
    dynamic_attributes: List[str] = Field(default_factory=list, description="Dynamic attributes found")
    
    # Content Analysis
    text_ratio: float = Field(0.0, description="Ratio of text to HTML")
    language: str = Field("", description="Detected language")
    keyword_density: Dict[str, int] = Field(default_factory=dict, description="Keyword density")
    structured_data: List[Dict[str, Any]] = Field(default_factory=list, description="Structured data found")
    
    # Technical Analysis
    frameworks: Dict[str, bool] = Field(default_factory=dict, description="Frameworks detected")
    apis: List[Dict[str, Any]] = Field(default_factory=list, description="APIs detected")
    pagination_patterns: List[str] = Field(default_factory=list, description="Pagination patterns")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    # SEO & Security
    seo_meta: Dict[str, str] = Field(default_factory=dict, description="SEO metadata")
    security_headers: Dict[str, Optional[str]] = Field(default_factory=dict, description="Security headers")
    robots_txt: str = Field("", description="Robots.txt content")
    
    # Enhanced Analysis
    content_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Content patterns detected")
    extraction_test_results: List[Dict[str, Any]] = Field(default_factory=list, description="Extraction test results")
    interactive_elements: List[Dict[str, Any]] = Field(default_factory=list, description="Interactive elements detected")
    visual_structure: Dict[str, Any] = Field(default_factory=dict, description="Visual structure analysis")
    selector_strategies: List[Dict[str, Any]] = Field(default_factory=list, description="Selector strategies")
    dom_visualization: List[str] = Field(default_factory=list, description="DOM structure visualization")
    code_templates: Dict[str, Any] = Field(default_factory=dict, description="Code templates information")
    
    # Recommendations
    selector_suggestions: List[str] = Field(default_factory=list, description="Selector suggestions")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    class Config:
        """Pydantic model configuration"""
        json_encoders = {
            # Custom JSON encoders for types that aren't JSON serializable
            set: list,  # Convert sets to lists
            # Add more custom encoders as needed
        }

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
    website_analysis: Optional[WebsiteAnalysis] = Field(None, description="Analysis of the website structure and content")