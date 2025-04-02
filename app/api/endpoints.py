from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from ..models.schemas import ScrapeRequest, ScrapeResponse, CodeExecutionResult, ScrapingIssue
from ..services.llm_service import LLMService
from ..services.scraper_helper import execute_code_wrapper
from ..utils.logger import app_logger, log_code_execution
import json
import os
from typing import Dict, List, Optional, Any

router = APIRouter()
llm_service = LLMService()

DEFAULT_MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))  # Default from environment

@router.post("/scrape", response_model=ScrapeResponse)
async def scrape_website(
    request: ScrapeRequest, 
    max_attempts: int = Query(DEFAULT_MAX_ATTEMPTS, description="Maximum number of code refinement attempts")
):
    """
    Endpoint to scrape a website using LLM-generated code with configurable attempts
    """
    app_logger.info(f"New scraping request for URL: {request.url}")
    app_logger.info(f"Model requested: {request.llm_model}")
    app_logger.info(f"Expected data: {request.expected_data}")
    app_logger.info(f"Maximum attempts: {max_attempts}")
    
    # Set up LLMs
    try:
        # Setup helper LLM with our API key from environment
        llm_service.setup_helper_llm()
        
        # Setup coding LLM with user-provided API key
        llm_service.setup_coding_llm(request.llm_model, request.api_key)
    except ValueError as e:
        error_msg = str(e)
        app_logger.error(f"LLM setup error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"Error setting up LLMs: {str(e)}"
        app_logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Format scraping prompt
    try:
        app_logger.info("Step 1: Formatting scraping prompt with helper LLM")
        formatted_prompt = llm_service.format_scraping_prompt(
            url=str(request.url),
            expected_data=request.expected_data
        )
    except Exception as e:
        error_msg = f"Error formatting prompt: {str(e)}"
        app_logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Execute scraping with retry logic
    attempts = 0
    execution_results = []
    code_success = False  # Code execution success
    data_success = False  # Data validation success
    final_code = None
    scraped_data = None
    validation_reason = ""
    
    app_logger.info(f"Beginning scraping attempts (max: {max_attempts})")
    
    while attempts < max_attempts:  # Continue until max attempts
        attempts += 1
        app_logger.info(f"Attempt {attempts}/{max_attempts}")
        
        try:
            # Generate scraping code
            app_logger.info("Step 2: Generating scraping code with coding LLM")
            raw_response = llm_service.generate_scraping_code(formatted_prompt)
            
            app_logger.info("Step 3: Extracting clean code with helper LLM")
            clean_code = llm_service.extract_code_from_response(raw_response)
            
            # Debug the code
            app_logger.info("Generated code content:")
            app_logger.info("-" * 80)
            app_logger.info(clean_code)
            app_logger.info("-" * 80)
            
            # Execute the code
            app_logger.info("Step 4: Executing the generated code")
            execution_result_dict = execute_code_wrapper(clean_code)
            
            # Convert the dictionary to a CodeExecutionResult object
            execution_result = CodeExecutionResult(
                stdout=execution_result_dict.get("stdout", ""),
                stderr=execution_result_dict.get("stderr", ""),
                success=execution_result_dict.get("success", False),
                execution_time=execution_result_dict.get("execution_time", 0.0),
                fix_methods=execution_result_dict.get("fix_methods", []),
                formatted_code=execution_result_dict.get("formatted_code", clean_code),
                scraping_issues=[
                    ScrapingIssue(
                        error_type=issue.get("error_type", "unknown"),
                        explanation=issue.get("explanation", "No explanation"),
                        recommendation=issue.get("recommendation", "No recommendation")
                    )
                    for issue in execution_result_dict.get("scraping_issues", [])
                ] if "scraping_issues" in execution_result_dict and execution_result_dict["scraping_issues"] else None,
                scraper_type=execution_result_dict.get("scraper_type", "unknown")
            )
            
            execution_results.append(execution_result)
            
            # Check if code execution was successful
            app_logger.info("Step 5: Checking if code execution was successful")
            code_success = execution_result.success and execution_result.stdout.strip()
            
            if code_success:
                app_logger.info("Code execution successful. Processing output...")
                final_code = clean_code
                
                # Process the output
                output_text = execution_result.stdout.strip()
                app_logger.info(f"Raw output: {output_text[:200]}...")
                
                # Log details for debugging
                app_logger.info(f"Stdout:\n{'-'*40}\n{output_text}\n{'-'*40}")
                
                # Parse the data
                try:
                    if output_text.startswith('{') or output_text.startswith('['):
                        scraped_data = json.loads(output_text)
                        app_logger.info("Output parsed as JSON successfully")
                    else:
                        # Look for JSON in the output
                        import re
                        json_match = re.search(r'(\{.*\}|\[.*\])', output_text, re.DOTALL)
                        if json_match:
                            try:
                                json_part = json_match.group(0)
                                scraped_data = json.loads(json_part)
                                app_logger.info("Found and parsed JSON within output")
                            except json.JSONDecodeError:
                                app_logger.info("Found potential JSON but couldn't parse it")
                                scraped_data = output_text
                        else:
                            # If not valid JSON, just use the raw output
                            app_logger.info("Output not JSON, using raw text")
                            scraped_data = output_text
                except json.JSONDecodeError:
                    # If not valid JSON, just use the raw output
                    app_logger.info("JSON parsing failed, using raw text")
                    scraped_data = output_text
                
                # Validate the extracted data against user requirements
                app_logger.info("Step 6: Validating extracted data against requirements")
                data_success, validation_reason = llm_service.validate_extracted_data(
                    scraped_data,
                    request.expected_data
                )
                
                if data_success:
                    app_logger.info("Data validation successful. Extracted data meets requirements.")
                    # We're done - data is valid!
                    break
                else:
                    app_logger.info(f"Data validation failed. Reason: {validation_reason}")
                    # Code executed, but data doesn't meet requirements
                    # Create a refined prompt specifically addressing the data extraction issue
                    app_logger.info("Step 7: Creating data-specific refinement prompt")
                    data_refinement_prompt = llm_service.create_data_refinement_prompt(
                        code=clean_code,
                        current_data=scraped_data,
                        expected_data=request.expected_data,
                        validation_reason=validation_reason
                    )
                    formatted_prompt = data_refinement_prompt
                    app_logger.info("Data refinement prompt created, continuing to next attempt")
                    
                    # Only exit the loop if we're on the last attempt
                    if attempts >= max_attempts:
                        break
            else:
                app_logger.info("Code execution failed. Refining code...")
                
                # Refine the code based on the error
                app_logger.info("Step 7: Refining code with error feedback")
                refined_prompt = llm_service.refine_code_with_error(
                    code=clean_code,
                    execution_result=execution_result
                )
                formatted_prompt = refined_prompt
                app_logger.info("Code refinement complete, continuing to next attempt")
                
                # Only exit the loop if we're on the last attempt
                if attempts >= max_attempts:
                    break
                    
        except Exception as e:
            error_msg = f"Error during execution attempt {attempts}: {str(e)}"
            app_logger.error(error_msg)
            execution_results.append(CodeExecutionResult(
                stdout="",
                stderr=error_msg,
                success=False,
                execution_time=0,
                fix_methods=[],
                scraping_issues=[]
            ))
            
            # If there's an exception, we should still continue to the next attempt
            # unless we've reached the maximum number of attempts
            if attempts >= max_attempts:
                app_logger.warning("Maximum attempts reached with errors, stopping.")
                break
    
    # Determine final success state - success only if both code execution and data validation succeeded
    final_success = code_success and data_success
    
    app_logger.info(f"Scraping process completed after {attempts} attempts.")
    app_logger.info(f"Code execution success: {code_success}")
    app_logger.info(f"Data validation success: {data_success}")
    if not data_success and validation_reason:
        app_logger.info(f"Data validation failed reason: {validation_reason}")
    
    # Return the results
    return ScrapeResponse(
        success=final_success,
        data=scraped_data,
        code=final_code,
        attempts=attempts,
        execution_results=execution_results
    )