import json
from typing import Dict, Any, Optional, Tuple, List
import os
import re
import time
from ..models.schemas import WebsiteAnalysis, CodeExecutionResult

def create_enhanced_scraping_prompt(url: str, expected_data: str, example_output: str, website_analysis) -> str:
    """
    Create an enhanced scraping prompt using the improved website analysis
    
    Args:
        url: Target URL
        expected_data: User's description of expected data
        example_output: Example output format JSON
        website_analysis: WebsiteAnalysis object
        
    Returns:
        Enhanced prompt string
    """
    # Base prompt
    prompt = f"""
    I need to scrape data from this website: {url}

    The specific data I want to extract is: {expected_data}
    """

    # Check if we have website analysis
    if website_analysis:
        # Extract key information from the analysis
        is_dynamic = getattr(website_analysis, 'is_dynamic', False)
        components = getattr(website_analysis, 'components', {})
        frameworks = getattr(website_analysis, 'frameworks', {})
        recommendations = getattr(website_analysis, 'recommendations', [])
        selector_suggestions = getattr(website_analysis, 'selector_suggestions', [])
        pagination_patterns = getattr(website_analysis, 'pagination_patterns', [])
        content_hierarchy = getattr(website_analysis, 'content_hierarchy', {})
        text_ratio = getattr(website_analysis, 'text_ratio', 0)
        language = getattr(website_analysis, 'language', '')
        keyword_density = getattr(website_analysis, 'keyword_density', {})
        apis = getattr(website_analysis, 'apis', [])
        performance_metrics = getattr(website_analysis, 'performance_metrics', {})
        security_headers = getattr(website_analysis, 'security_headers', {})
        dynamic_attributes = getattr(website_analysis, 'dynamic_attributes', [])
        
        # Enhanced analysis features
        content_patterns = getattr(website_analysis, 'content_patterns', [])
        extraction_test_results = getattr(website_analysis, 'extraction_test_results', [])
        interactive_elements = getattr(website_analysis, 'interactive_elements', [])
        selector_strategies = getattr(website_analysis, 'selector_strategies', [])
        code_templates = getattr(website_analysis, 'code_templates', {})

        # Format the analysis information
        analysis_info = f"""

        WEBSITE ANALYSIS RESULTS:
        - Dynamic content: {'Yes' if is_dynamic else 'No'}
        - Components detected: {', '.join([f"{k} ({v})" for k, v in components.items() if v > 0])}
        - Frameworks: {', '.join([k for k, v in frameworks.items() if v]) or 'None detected'}
        - Text ratio: {text_ratio:.2f}
        - Language: {language if language else 'Not detected'}
        - Content structure: {', '.join([f"{k} ({v})" for k, v in content_hierarchy.items() if v > 0])}
        """

        # Add recommendations if available
        if recommendations:
            analysis_info += f"""
        - Technical recommendations:
          * {"\n          * ".join(recommendations[:5])}
        """

        # Add selector suggestions if available
        if selector_suggestions:
            analysis_info += f"""
        - Suggested selectors:
          * {"\n          * ".join(selector_suggestions[:3])}
        """
        
        # Add content patterns if available (enhanced)
        if content_patterns:
            sorted_patterns = sorted(content_patterns, key=lambda x: x.get('count', 0) if isinstance(x, dict) else 0, reverse=True)
            top_patterns = sorted_patterns[:3]
            
            pattern_info = []
            for p in top_patterns:
                if isinstance(p, dict):
                    pattern_info.append(f"{p.get('child_selector', '')} ({p.get('count', 0)} items, {p.get('confidence', 'medium')} confidence)")
            
            if pattern_info:
                analysis_info += f"""
        - Content patterns detected:
          * {"\n          * ".join(pattern_info)}
        """
            
        # Add extraction test results if available (enhanced)
        if extraction_test_results:
            # Find the entry with the highest count
            best_extraction = None
            highest_count = 0
            
            for result in extraction_test_results:
                if isinstance(result, dict) and result.get('count', 0) > highest_count:
                    highest_count = result.get('count', 0)
                    best_extraction = result
            
            if best_extraction:
                analysis_info += f"""
        - Best extraction pattern: {best_extraction.get('pattern', 'unknown')} using '{best_extraction.get('selector', '')}' found {best_extraction.get('count', 0)} items
        """
                
        # Add interactive elements if available (enhanced)
        if interactive_elements:
            button_elements = [el for el in interactive_elements if isinstance(el, dict) and el.get('type') == 'button']
            pagination_elements = [el for el in interactive_elements if isinstance(el, dict) and el.get('type') == 'pagination']
            
            if button_elements:
                analysis_info += f"""
        - Interactive buttons detected: {len(button_elements)} (e.g., {button_elements[0].get('text', '') if button_elements else ''})
        """
            if pagination_elements:
                analysis_info += f"""
        - Pagination elements detected: {len(pagination_elements)}
        """

        # Add pagination information if available
        if pagination_patterns:
            analysis_info += f"""
        - Pagination detected: {', '.join(pagination_patterns[:3])}
        """

        # Add dynamic attributes if available
        if dynamic_attributes:
            analysis_info += f"""
        - Dynamic attributes: {', '.join(dynamic_attributes[:5])}{'...' if len(dynamic_attributes) > 5 else ''}
        """

        # Add API information if available
        if apis:
            analysis_info += f"""
        - APIs detected: {len(apis)}
        """

        # Add performance metrics if available
        if performance_metrics:
            load_time = performance_metrics.get('loadTime', 0)
            analysis_info += f"""
        - Load time: {load_time}ms
        """

        # Add keyword information if available
        if keyword_density:
            top_keywords = list(keyword_density.items())[:5]
            analysis_info += f"""
        - Top keywords: {', '.join([f"{k} ({v})" for k, v in top_keywords])}
        """

        # Add security information if available
        if any(security_headers.values()):
            analysis_info += f"""
        - Security headers: {', '.join([k for k, v in security_headers.items() if v])}
        """
        
        # Add code template information if available (enhanced)
        if code_templates and isinstance(code_templates, dict) and 'selectors' in code_templates and code_templates['selectors']:
            selectors = code_templates['selectors']
            selector_info = [f"{k}: {v}" for k, v in selectors.items()]
            if selector_info:
                analysis_info += f"""
        - Recommended selectors for extraction:
          * {"\n          * ".join(selector_info)}
        """

        # Add the analysis to the prompt
        prompt += analysis_info

    # Add the rest of the prompt - guidance for scraping
    if is_dynamic:
        prompt += f"""
        
        IMPORTANT TECHNICAL CONSIDERATIONS:
        - This website uses dynamic content that requires JavaScript to render. Use Playwright or Selenium.
        - Wait for content to load properly before extracting data.
        - Consider implementing explicit waits for key elements.
        """
    else:
        prompt += f"""
        
        TECHNICAL CONSIDERATIONS:
        - This website appears to be mostly static content. A simple requests + BeautifulSoup approach should work.
        - Make sure to set appropriate headers including User-Agent.
        """
        
    # Add code suggestions based on analysis
    if content_patterns or extraction_test_results:
        prompt += f"""
        
        CODE SUGGESTIONS:
        - Target the main content container first, then extract individual items.
        - Use robust selector strategies that can handle minor page changes.
        - Implement proper error handling for missing elements.
        """
        
    # Add specific implementation tips for the target data
    prompt += f"""
        
        I need you to format this request into a clear, detailed prompt for an LLM that will
        generate Python web scraping code. The prompt should be structured to help the LLM
        understand exactly what data to extract and how to handle the website.

        Format the prompt to include:
        - Clear description of the target website
        - Specific data fields to extract
        - Any special considerations for this website based on the analysis
        - Request for robust error handling
        - Request for clear output formatting

        IMPORTANT: The code should print the output as JSON. Here's an example format, but the model should adapt this
        to best represent the specific data being extracted:
        ```
        {example_output}
        ```

        The model has flexibility to design the JSON structure that best represents the extracted data,
        as long as it's well-organized and contains all the requested information.

        Make sure to emphasize that the code should:
        1. Be complete and runnable Python code
        2. Include all necessary imports
        3. Print the results as shown in the example format above
        4. Use the appropriate scraping approach based on the website analysis
        """
        
    return prompt