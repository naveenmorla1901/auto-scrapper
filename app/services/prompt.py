"""
Prompt templates for different stages of the auto-scraping process.
"""

SCRAPE_REQUEST_TEMPLATE = """
I need to scrape data from the website: {url}

The specific data I want to extract is:
{expected_data}

Please generate complete, working Python code to scrape this data. The code should:

1. Handle different potential website structures and behaviors
2. Include appropriate error handling 
3. Output the data in a structured, clean format
4. Use appropriate libraries (requests, BeautifulSoup, Selenium, or Playwright) based on the website's complexity
5. Follow ethical scraping practices (respect robots.txt, reasonable delays, etc.)

Return only the executable Python code without explanations.
"""

ERROR_REFINEMENT_TEMPLATE = """
The following Python web scraping code encountered errors:

CODE:
```python
{code}
```

ERROR OUTPUT:
{error}

EXECUTION DETAILS:
- Success: {success}
- Execution Time: {execution_time} seconds
- Scraper Type: {scraper_type}

Please fix this code to address the following issues:
{issues}

Suggested fixes:
{fix_methods}

Return only the complete, corrected Python code without explanations.
"""

EXTRACTION_VERIFICATION_TEMPLATE = """
Analyze the following output from a web scraping operation:

OUTPUT:
{output}

ERROR (if any):
{error}

Does this output contain the requested data: {expected_data}?
Answer only YES or NO.
"""

CODE_EXTRACTION_TEMPLATE = """
Extract only the complete, executable Python code from the following LLM response.
Return just the Python code with no markdown formatting, explanations, or other text.

LLM RESPONSE:
{llm_response}
"""

DATA_EXTRACTION_TEMPLATE = """
The following is the output from a web scraping operation:

{output}

Please extract and format the meaningful data from this output in a clean, structured way.
Format the result as valid JSON if possible.
"""