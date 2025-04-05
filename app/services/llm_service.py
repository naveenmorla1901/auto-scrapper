import json
from typing import Dict, Any, Optional, Tuple
import os
import re
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from ..models.schemas import CodeExecutionResult, TokenUsage
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

        # Token usage tracking
        self.helper_token_usage = TokenUsage()
        self.coding_token_usage = TokenUsage()

        # Model pricing (per 1000 tokens, in USD)
        self.pricing = {
            # OpenAI models
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.0015, "output": 0.006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},

            # Google models
            "gemini-1.5-pro": {"input": 0.0025, "output": 0.0075},
            "gemini-1.5-flash": {"input": 0.0005, "output": 0.0015},
            "gemini-2.0-flash-lite": {"input": 0.0001, "output": 0.0003},
            "gemini-1.0-pro": {"input": 0.0025, "output": 0.0075},
            "gemini-1.0-ultra": {"input": 0.0025, "output": 0.0075},

            # Anthropic models
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},

            # Mistral models
            "mistral-large": {"input": 0.002, "output": 0.006},
            "mistral-medium": {"input": 0.0008, "output": 0.0024},
            "mistral-small": {"input": 0.0002, "output": 0.0006},

            # Default fallback pricing
            "default": {"input": 0.001, "output": 0.002}
        }

        # Model provider mapping
        self.provider_patterns = {
            "openai": r"^gpt-",
            "google": r"^gemini-",
            "anthropic": r"^claude-",
            "mistral": r"^mistral-",
            "meta": r"^llama-",
            "deepseek": r"^deepseek-",
            "cohere": r"^command",
            "ai21": r"^j2-",
            "together": r"^(yi-|qwen|falcon)"
        }

        # Map of user-friendly model names to actual API model names
        # This is especially important for models with version-specific API names
        self.model_name_map = {
            # Claude models - their actual API names
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",

            # Mistral models - their actual API names
            "mistral-large": "mistral-large-latest",
            "mistral-medium": "mistral-medium-latest",
            "mistral-small": "mistral-small-latest",

            # DeepSeek models - their actual API names
            "deepseek-coder": "deepseek-coder-33b-instruct",
            "deepseek-chat": "deepseek-llm-67b-chat",
            "deepseek-llm-67b": "deepseek-llm-67b-base",
            "deepseek-llm-7b": "deepseek-llm-7b-base",

            # Cohere models
            "command-r": "command-r-v1",
            "command-r-plus": "command-r-plus-v1",
            "command": "command-light-v1",

            # Together AI hosted models
            "yi-34b": "01-ai/Yi-34B-Chat",
            "qwen-72b": "Qwen/Qwen-72B-Chat",
            "falcon-180b": "tiiuae/falcon-180B-chat",

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
                try:
                    # Access through Together AI
                    from langchain_community.chat_models import ChatTogetherAI
                    self.coding_llm = ChatTogetherAI(
                        model=self.map_model_name(model),
                        together_api_key=api_key,
                        temperature=0.1
                    )
                except ImportError:
                    error_msg = "Meta Llama support requires langchain-community package. Install it with: pip install langchain-community"
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)
            elif provider == "deepseek":
                try:
                    # DeepSeek models can be accessed through Together AI
                    from langchain_community.chat_models import ChatTogetherAI
                    self.coding_llm = ChatTogetherAI(
                        model=self.map_model_name(model),
                        together_api_key=api_key,
                        temperature=0.1
                    )
                except ImportError:
                    error_msg = "DeepSeek support requires langchain-community package. Install it with: pip install langchain-community"
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)
            elif provider == "cohere":
                try:
                    # Try using the community implementation first
                    try:
                        from langchain_community.chat_models import ChatCohere
                        self.coding_llm = ChatCohere(
                            model=self.map_model_name(model),
                            cohere_api_key=api_key,
                            temperature=0.1
                        )
                    except (ImportError, AttributeError):
                        # Fall back to dedicated package if available
                        from langchain_cohere import ChatCohere
                        self.coding_llm = ChatCohere(
                            model=self.map_model_name(model),
                            cohere_api_key=api_key,
                            temperature=0.1
                        )
                except ImportError:
                    error_msg = "Cohere support requires langchain-community or langchain-cohere package. Install with: pip install langchain-community langchain-cohere"
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)
            elif provider == "ai21":
                try:
                    from langchain_community.chat_models import ChatAI21
                    self.coding_llm = ChatAI21(
                        model=self.map_model_name(model),
                        ai21_api_key=api_key,
                        temperature=0.1
                    )
                except ImportError:
                    error_msg = "AI21 support requires langchain-community package. Install it with: pip install langchain-community"
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)
            elif provider == "together":
                try:
                    from langchain_community.chat_models import ChatTogetherAI
                    self.coding_llm = ChatTogetherAI(
                        model=self.map_model_name(model),
                        together_api_key=api_key,
                        temperature=0.1
                    )
                except ImportError:
                    error_msg = "Together AI support requires langchain-community package. Install it with: pip install langchain-community"
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

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string"""
        # Simple estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of a model call based on token usage"""
        # Get pricing for the model, or use default if not found
        pricing = self.pricing.get(model, self.pricing["default"])

        # Calculate cost (convert from per 1000 tokens to per token)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost

    def update_token_usage(self, is_helper: bool, input_text: str, output_text: str) -> TokenUsage:
        """Update token usage for a model call"""
        # Determine which usage to update
        usage = self.helper_token_usage if is_helper else self.coding_token_usage
        model = self.helper_model_name if is_helper else self.coding_model_name

        # Estimate tokens
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)

        # Update usage
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.total_tokens = usage.input_tokens + usage.output_tokens

        # Calculate cost
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        usage.cost += cost

        return usage

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

        # Track token usage
        start_time = time.time()

        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)

        # Update token usage
        self.update_token_usage(True, prompt, response.content)

        # Log execution time
        execution_time = time.time() - start_time
        app_logger.info(f"Helper LLM response time: {execution_time:.2f} seconds")

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

        # Track token usage
        start_time = time.time()

        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)

        # Update token usage
        self.update_token_usage(True, prompt, response.content)

        # Log execution time
        execution_time = time.time() - start_time
        app_logger.info(f"Helper LLM code extraction time: {execution_time:.2f} seconds")

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

        # Track token usage
        start_time = time.time()

        response = self.coding_llm.invoke(messages)

        # Update token usage (include system message in input)
        self.update_token_usage(False, system_message + formatted_prompt, response.content)

        # Log execution time
        execution_time = time.time() - start_time
        app_logger.info(f"Coding LLM response time: {execution_time:.2f} seconds")

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

        # Track token usage
        start_time = time.time()

        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)

        # Update token usage
        self.update_token_usage(True, prompt, response.content)

        # Log execution time
        execution_time = time.time() - start_time
        app_logger.info(f"Helper LLM refinement time: {execution_time:.2f} seconds")

        log_llm_interaction("OUTPUT", self.helper_model_name, response=response.content)
        return response.content

    def validate_extracted_data(self, data: Any, expected_data_description: str) -> Tuple[bool, str]:
        """
        Use helper LLM to validate if the extracted data matches user requirements

        Args:
            data: The extracted data (can be any type)
            expected_data_description: User's description of what they wanted

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not self.helper_llm:
            self.setup_helper_llm()

        # Convert data to a string representation
        data_str = str(data)
        if isinstance(data, (dict, list)):
            try:
                data_str = json.dumps(data, indent=2)
            except:
                data_str = str(data)

        prompt = f"""
        I need to validate if the extracted data meets the user's requirements.

        USER REQUESTED DATA:
        {expected_data_description}

        EXTRACTED DATA:
        {data_str}

        Analyze if the extracted data satisfies what the user was looking for.
        Consider:
        1. Does it contain the specific information types requested?
        2. Is it in a usable format?
        3. Is it complete or at least contains meaningful information?

        Respond with ONLY "YES" if the data satisfies the requirements or "NO" if it doesn't.
        If responding "NO", add a brief reason after, like "NO - Missing price information"
        """

        app_logger.info("Validating extracted data against requirements")
        log_llm_interaction("INPUT", self.helper_model_name, prompt)

        # Track token usage
        start_time = time.time()

        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)

        # Update token usage
        self.update_token_usage(True, prompt, response.content)

        # Log execution time
        execution_time = time.time() - start_time
        app_logger.info(f"Helper LLM validation time: {execution_time:.2f} seconds")

        validation_result = response.content.strip().upper()
        is_valid = validation_result.startswith("YES")

        app_logger.info(f"Data validation result: {'VALID' if is_valid else 'INVALID'}")
        log_llm_interaction("OUTPUT", self.helper_model_name, response=response.content)

        # Extract reason if validation failed
        reason = ""
        if not is_valid and "-" in validation_result:
            parts = validation_result.split("-", 1)
            if len(parts) > 1:
                reason = parts[1].strip()

        return is_valid, reason

    def create_data_refinement_prompt(self, code: str, current_data: Any, expected_data: str, validation_reason: str) -> str:
        """
        Create a prompt to refine code when extracted data doesn't meet requirements

        Args:
            code: The current code that executed successfully
            current_data: The data that was extracted (but doesn't meet requirements)
            expected_data: User's description of what they wanted
            validation_reason: Reason why validation failed

        Returns:
            Refined prompt for the coding LLM
        """
        if not self.helper_llm:
            self.setup_helper_llm()

        # Convert data to a string representation
        data_str = str(current_data)
        if isinstance(current_data, (dict, list)):
            try:
                data_str = json.dumps(current_data, indent=2)
            except:
                data_str = str(current_data)

        prompt = f"""
        The Python web scraping code executed successfully, but the extracted data doesn't meet the requirements.

        CODE:
        ```python
        {code}
        ```

        EXTRACTED DATA:
        {data_str}

        EXPECTED DATA DESCRIPTION:
        {expected_data}

        VALIDATION ISSUE:
        {validation_reason}

        Create a detailed prompt for an LLM to revise this code. The prompt should:
        1. Explain that the code executes but doesn't extract the right data
        2. Highlight specifically what data is missing or incorrect based on the validation issue
        3. Suggest how to modify the code to correctly extract the required data
        4. Request a complete, corrected version of the code that properly extracts and formats the data

        Focus on the data extraction logic, selectors, and output formatting.
        """

        app_logger.info("Creating data-specific refinement prompt")
        log_llm_interaction("INPUT", self.helper_model_name, prompt)

        # Track token usage
        start_time = time.time()

        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)

        # Update token usage
        self.update_token_usage(True, prompt, response.content)

        # Log execution time
        execution_time = time.time() - start_time
        app_logger.info(f"Helper LLM refinement prompt time: {execution_time:.2f} seconds")

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