"""
Helper functions to wrap the original scraper to handle dictionary conversion.
"""
from typing import Dict, Any, List
import logging
import json
import re
from ..utils.logger import app_logger

def generate_example_output(data_description: str) -> str:
    """
    Generate an example output format based on the data description

    Args:
        data_description: Description of the data to extract

    Returns:
        Example output format as a string
    """
    # Common data types to look for
    data_types = {
        "product": {
            "name": "Example Product",
            "price": "$19.99",
            "description": "This is a sample product description.",
            "rating": 4.5,
            "reviews": 120
        },
        "article": {
            "title": "Example Article Title",
            "author": "John Doe",
            "date": "2023-01-01",
            "content": "This is a sample article content."
        },
        "news": {
            "headline": "Breaking News Example",
            "summary": "This is a sample news summary.",
            "date": "2023-01-01",
            "source": "Example News Source"
        },
        "review": {
            "reviewer": "Jane Smith",
            "rating": 4.5,
            "date": "2023-01-01",
            "content": "This is a sample review content."
        },
        "profile": {
            "name": "Jane Doe",
            "username": "janedoe123",
            "bio": "This is a sample bio.",
            "followers": 1250,
            "following": 450
        },
        "comment": {
            "author": "John Smith",
            "date": "2023-01-01",
            "content": "This is a sample comment content.",
            "likes": 15
        },
        "post": {
            "author": "Jane Doe",
            "date": "2023-01-01",
            "content": "This is a sample post content.",
            "likes": 150,
            "comments": 25
        },
        "job": {
            "title": "Software Engineer",
            "company": "Example Corp",
            "location": "New York, NY",
            "salary": "$100,000 - $150,000",
            "description": "This is a sample job description."
        },
        "experience": {
            "title": "Senior Developer",
            "company": "Tech Company",
            "period": "2020-2023",
            "description": "Led development of key features."
        },
        "education": {
            "degree": "Master of Science",
            "institution": "Example University",
            "year": "2018",
            "field": "Computer Science"
        },
        "project": {
            "name": "Portfolio Website",
            "technologies": ["HTML", "CSS", "JavaScript"],
            "description": "Personal portfolio showcasing projects.",
            "link": "https://example.com"
        }
    }

    # Try to match the data description with a known data type
    description_lower = data_description.lower()
    matched_types = []

    for data_type in data_types:
        if data_type in description_lower:
            matched_types.append(data_type)

    # Check for specific keywords that might indicate certain data types
    if "journey" in description_lower or "career" in description_lower:
        if "experience" not in matched_types:
            matched_types.append("experience")

    if "education" in description_lower or "degree" in description_lower or "university" in description_lower:
        if "education" not in matched_types:
            matched_types.append("education")

    if "project" in description_lower or "portfolio" in description_lower:
        if "project" not in matched_types:
            matched_types.append("project")

    # If no match found, generate a generic example
    if not matched_types:
        app_logger.info(f"No specific data type matched for: {data_description}. Using generic example.")
        return json.dumps({
            "results": [
                {
                    "title": "Example Item",
                    "description": "This is a sample description.",
                    "value": "Sample value"
                }
            ]
        }, indent=2)

    # Create example output based on the matched data types
    example_data = {"results": []}

    # Add examples for each matched type (up to 3)
    for data_type in matched_types[:3]:
        example_data["results"].append(data_types[data_type])

    # If multiple items are expected, add another example of the first type
    if ("list" in description_lower or "multiple" in description_lower) and len(example_data["results"]) < 2:
        example_data["results"].append(data_types[matched_types[0]])

    app_logger.info(f"Generated example output for data types: {', '.join(matched_types)}")
    return json.dumps(example_data, indent=2)

def test_code_format(code_string: str) -> bool:
    """
    Test if the code string has proper Python format

    Args:
        code_string: Python code to test

    Returns:
        True if the code appears to be properly formatted Python code
    """
    # Check for basic Python syntax patterns
    import_pattern = r'^\s*(import|from)\s+\w+'
    has_imports = bool(re.search(import_pattern, code_string, re.MULTILINE))

    # Check for other Python patterns
    function_pattern = r'^\s*def\s+\w+'
    class_pattern = r'^\s*class\s+\w+'
    variable_pattern = r'^\s*\w+\s*='

    has_functions = bool(re.search(function_pattern, code_string, re.MULTILINE))
    has_classes = bool(re.search(class_pattern, code_string, re.MULTILINE))
    has_variables = bool(re.search(variable_pattern, code_string, re.MULTILINE))

    # Code should have at least imports and some form of code structure
    return has_imports and (has_functions or has_classes or has_variables)

def execute_code_wrapper(code_string: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Wrapper around the execute_code function to ensure it returns a dict with the right attributes

    Args:
        code_string: The Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results including success flag
    """
    # Check if the code looks like valid Python
    if not test_code_format(code_string):
        app_logger.error("Code doesn't appear to be valid Python")
        return {
            "success": False,
            "stdout": "",
            "stderr": "Code doesn't appear to be valid Python. Please check the format.",
            "execution_time": 0.0,
            "fix_methods": [],
            "scraping_issues": []
        }

    # Import the original function
    from ..services.scraper import execute_code

    # Log the code being executed
    app_logger.info("About to execute code:")
    code_lines = code_string.split('\n')
    for i, line in enumerate(code_lines):
        app_logger.info(f"{i+1:3d}: {line}")

    try:
        # Execute the code
        result = execute_code(code_string, timeout)

        # Ensure the result is a dictionary with the right attributes
        if not isinstance(result, dict):
            app_logger.error(f"execute_code returned non-dict: {type(result)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Internal error: execute_code returned non-dict type: {type(result)}",
                "execution_time": 0.0,
                "fix_methods": [],
                "scraping_issues": []
            }

        # Check for success key
        if "success" not in result:
            app_logger.warning("Adding missing 'success' key to execute_code result")
            result["success"] = False if result.get("stderr", "") else True

        return result
    except Exception as e:
        app_logger.error(f"Error in execute_code_wrapper: {str(e)}")
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Exception in execute_code_wrapper: {str(e)}",
            "execution_time": 0.0,
            "fix_methods": [],
            "scraping_issues": []
        }