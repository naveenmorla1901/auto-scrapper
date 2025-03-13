from typing import Dict, Any, Optional
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from ..models.schemas import CodeExecutionResult
from ..utils.logger import app_logger, log_llm_interaction

class LLMService:
    def __init__(self):
        """Initialize LLM service with default helper LLM"""
        # Default helper LLM - Gemini 2.0 Flash-Lite
        self.helper_llm = None
        self.coding_llm = None
        
        # Get API key for helper model from environment
        self.helper_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.helper_api_key:
            app_logger.warning("GOOGLE_API_KEY not found in environment variables")
        
        # Track current models
        self.helper_model_name = "gemini-2.0-flash-lite"
        self.coding_model_name = None
        
        # Model provider mapping
        self.provider_patterns = {
            "openai": r"^gpt-",
            "google": r"^gemini-",
            "anthropic": r"^claude-",
            "mistral": r"^mistral-",
            "meta": r"^llama-"
        }
        
        # Map of user-friendly model names to actual API model names
        # This is especially important for Claude models
        self.model_name_map = {
            # Claude models - their actual API names
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",  # Fictional future model, just a placeholder
            
            # The rest use the same names
            "default": lambda x: x  # Default is to use the same name
        }
        
    def map_model_name(self, model: str) -> str:
        """Map a user-friendly model name to the actual API model name"""
        # Check if there's a specific mapping for this model
        if model in self.model_name_map:
            return self.model_name_map[model]
        # Otherwise, return the model name as is
        return model
        
    def get_provider(self, model: str) -> str:
        """Determine the provider based on the model name"""
        for provider, pattern in self.provider_patterns.items():
            if re.match(pattern, model.lower()):
                return provider
        raise ValueError(f"Unable to determine provider for model: {model}")
        
    def setup_helper_llm(self):
        """Set up the helper LLM (Gemini 2.0 Flash-Lite) using our API key"""
        if not self.helper_api_key:
            raise ValueError("No Google API key found for helper LLM. Please set GOOGLE_API_KEY in .env file.")
        
        app_logger.info(f"Setting up helper LLM: {self.helper_model_name}")
        
        self.helper_llm = ChatGoogleGenerativeAI(
            model=self.helper_model_name,
            temperature=0.2,
            google_api_key=self.helper_api_key
        )
        
    def setup_coding_llm(self, model: str, api_key: str):
        """Set up the coding LLM based on user preference and API key"""
        app_logger.info(f"Setting up coding LLM: {model}")
        self.coding_model_name = model
        
        try:
            provider = self.get_provider(model)
            app_logger.info(f"Detected provider: {provider}")
            
            if provider == "openai":
                self.coding_llm = ChatOpenAI(
                    model=model,
                    temperature=0.1,
                    openai_api_key=api_key
                )
            elif provider == "google":
                self.coding_llm = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=0.1,
                    google_api_key=api_key
                )
            elif provider == "anthropic":
                try:
                    from langchain_anthropic import ChatAnthropic
                    
                    # Use the mapped model name for Claude models
                    api_model_name = self.map_model_name(model)
                    app_logger.info(f"Using Claude API model name: {api_model_name}")
                    
                    self.coding_llm = ChatAnthropic(
                        model=api_model_name,
                        temperature=0.1,
                        anthropic_api_key=api_key
                    )
                except ImportError:
                    error_msg = "Anthropic support requires langchain-anthropic package. Install it with: pip install langchain-anthropic"
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)
            elif provider == "mistral":
                try:
                    from langchain_mistralai.chat_models import ChatMistralAI
                    self.coding_llm = ChatMistralAI(
                        model=model,
                        temperature=0.1,
                        mistral_api_key=api_key
                    )
                except ImportError:
                    error_msg = "Mistral AI support requires langchain-mistralai package. Install it with: pip install langchain-mistralai"
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)
            elif provider == "meta":
                # For Meta's Llama models, typically accessed through hosted APIs
                # This is just a placeholder - adjust based on how you access Llama models
                try:
                    # For example, through replicate
                    from langchain_community.llms import Replicate
                    self.coding_llm = Replicate(
                        model=f"meta/{model}",
                        replicate_api_token=api_key
                    )
                except ImportError:
                    error_msg = "Meta Llama support requires specific packages. See documentation."
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)
            else:
                error_msg = f"Unsupported provider: {provider}"
                app_logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            error_msg = f"Error setting up {model}: {str(e)}"
            app_logger.error(error_msg)
            raise ValueError(error_msg)
    
    def format_scraping_prompt(self, url: str, expected_data: str) -> str:
        """Use helper LLM to format the scraping prompt"""
        if not self.helper_llm:
            self.setup_helper_llm()
        
        # Generate an example output format based on the expected data
        from ..services.scraper_helper import generate_example_output
        example_output = generate_example_output(expected_data)
            
        prompt = f"""
        I need to scrape data from this website: {url}
        
        The specific data I want to extract is: {expected_data}
        
        I need you to format this request into a clear, detailed prompt for an LLM that will 
        generate Python web scraping code. The prompt should be structured to help the LLM 
        understand exactly what data to extract and how to handle the website.
        
        Format the prompt to include:
        - Clear description of the target website
        - Specific data fields to extract
        - Any special considerations for this website
        - Request for robust error handling
        - Request for clear output formatting
        
        IMPORTANT: The code should print the output in this format:
        ```
        {example_output}
        ```
        
        Make sure to emphasize that the code should:
        1. Be complete and runnable Python code
        2. Include all necessary imports
        3. Print the results as shown in the example format above
        """
        
        app_logger.info(f"Formatting scraping prompt for URL: {url}")
        log_llm_interaction("INPUT", self.helper_model_name, prompt)
        
        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)
        
        log_llm_interaction("OUTPUT", self.helper_model_name, response=response.content)
        return response.content
    
    def extract_code_from_response(self, response: str) -> str:
        """Use helper LLM to extract clean code from the coding LLM's response"""
        if not self.helper_llm:
            self.setup_helper_llm()
        
        # First, try to extract code using simple regex for markdown code blocks
        import re
        python_code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if python_code_blocks:
            app_logger.info("Found Python code block using regex")
            return python_code_blocks[0].strip()
            
        # If no Python blocks, look for any code blocks
        code_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if code_blocks:
            app_logger.info("Found generic code block using regex")
            return code_blocks[0].strip()
        
        # If regex fails, use the helper LLM as a backup method
        prompt = f"""
        Extract only the Python code from the following LLM response. 
        Return just the complete, executable Python code for web scraping, nothing else.
        Include all necessary imports.
        Make sure the code is complete and not truncated.
        Do not include any markdown formatting, backticks, or explanation text.
        
        LLM Response:
        {response}
        """
        
        app_logger.info("Using helper LLM to extract code (regex methods failed)")
        log_llm_interaction("INPUT", self.helper_model_name, prompt)
        
        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)
        
        extracted_code = response.content.strip()
        
        # Clean up any remaining markdown artifacts
        extracted_code = extracted_code.replace("```python", "").replace("```", "").strip()
        
        log_llm_interaction("OUTPUT", self.helper_model_name, response=extracted_code)
        
        # Check if the extracted code starts with import statements or looks like valid Python
        if not (extracted_code.startswith("import") or 
                extracted_code.startswith("from") or 
                "def" in extracted_code or 
                "class" in extracted_code):
            app_logger.warning("Extracted code doesn't look like valid Python")
            app_logger.warning("-" * 80)
            app_logger.warning(extracted_code[:200])
            app_logger.warning("-" * 80)
        
        return extracted_code
    
    def generate_scraping_code(self, formatted_prompt: str) -> str:
        """Use coding LLM to generate scraping code"""
        if not self.coding_llm:
            error_msg = "Coding LLM not set up. Call setup_coding_llm first."
            app_logger.error(error_msg)
            raise ValueError(error_msg)
            
        system_message = """
        You are a focused Python web scraper developer. Your task is to write code that extracts 
        the specified data from a website.
        
        CRITICAL REQUIREMENTS:
        - ONLY return Python code, no explanations
        - Format your entire response as a code block with ```python at the start and ``` at the end
        - Include all necessary imports at the top
        - Use requests and BeautifulSoup for simple sites
        - Print the extracted data at the end (preferably as JSON)
        - Handle basic errors in case the site structure changes
        
        SIMPLICITY IS KEY:
        - Focus on extracting the data requested
        - Don't worry about perfect error handling
        - Don't add complex features
        
        Example of correct response format:
        ```python
        import requests
        from bs4 import BeautifulSoup
        import json
        
        url = "https://example.com"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract the data
        data = {"results": ["item1", "item2"]}
        
        # Print the result
        print(json.dumps(data, indent=2))
        ```
        """
        
        app_logger.info(f"Generating scraping code with {self.coding_model_name}")
        log_llm_interaction("INPUT", self.coding_model_name, f"System: {system_message}\n\nUser: {formatted_prompt}")
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=formatted_prompt)
        ]
        
        response = self.coding_llm.invoke(messages)
        
        log_llm_interaction("OUTPUT", self.coding_model_name, response=response.content)
        return response.content
    
    def refine_code_with_error(self, code: str, execution_result: CodeExecutionResult) -> str:
        """Use helper LLM to generate a prompt for refining code based on execution errors"""
        if not self.helper_llm:
            self.setup_helper_llm()
        
        # Extract specific error details
        error_details = ""
        if execution_result.stderr:
            error_details = execution_result.stderr[:1000]  # Limit to 1000 chars to avoid token limits
        
        # Extract any scraping issues identified
        scraping_issues = ""
        if execution_result.scraping_issues:
            for i, issue in enumerate(execution_result.scraping_issues):
                scraping_issues += f"{i+1}. {issue.error_type}: {issue.explanation}\n   Recommendation: {issue.recommendation}\n"
        
        # Check for syntax errors specifically
        has_syntax_error = "syntax" in error_details.lower() or "SyntaxError" in error_details
        
        prompt = f"""
        I need to fix this Python web scraping code that failed to run correctly.
        
        CODE:
        ```python
        {code}
        ```
        
        EXECUTION ISSUES:
        - Success: {execution_result.success}
        - Error output: {error_details}
        - Execution time: {execution_result.execution_time} seconds
        
        {f"SCRAPING ISSUES IDENTIFIED:\n{scraping_issues}" if scraping_issues else ""}
        
        Create a detailed prompt for an LLM to fix this code. Be very specific about:
        
        1. Exactly what went wrong (syntax errors, connection issues, selector problems, etc.)
        2. What specific parts of the code need to be fixed
        3. How to properly extract and print the data
        
        {"IMPORTANT: The code appears to have syntax errors. Focus primarily on fixing the basic syntax." if has_syntax_error else ""}
        
        The revised code should:
        - Import all necessary libraries
        - Use robust error handling
        - Print the results in a clear format (preferably JSON)
        - Include proper User-Agent headers
        - Be complete and executable
        
        Format this as a prompt that clearly identifies the issues and provides specific guidance on how to fix them.
        """
        
        app_logger.info("Refining code with error feedback")
        log_llm_interaction("INPUT", self.helper_model_name, prompt)
        
        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)
        
        log_llm_interaction("OUTPUT", self.helper_model_name, response=response.content)
        return response.content
    
    def check_success(self, execution_result: CodeExecutionResult) -> bool:
        """Use helper LLM to determine if the scraping was successful based on the output"""
        if not self.helper_llm:
            self.setup_helper_llm()
            
        prompt = f"""
        Analyze these execution results and determine if the web scraping was successful.
        A successful scraping should have:
        1. No critical errors in stderr
        2. Meaningful data in stdout
        3. The 'success' flag set to True
        
        EXECUTION RESULT:
        Success flag: {execution_result.success}
        Stdout: {execution_result.stdout}
        Stderr: {execution_result.stderr}
        Execution time: {execution_result.execution_time} seconds
        
        Respond with ONLY "YES" if successful or "NO" if not successful.
        """
        
        app_logger.info("Checking if scraping was successful")
        log_llm_interaction("INPUT", self.helper_model_name, prompt)
        
        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)
        
        is_success = "YES" in response.content.upper()
        app_logger.info(f"Scraping success check result: {'SUCCESS' if is_success else 'FAILURE'}")
        log_llm_interaction("OUTPUT", self.helper_model_name, response=response.content)
        
        return is_success