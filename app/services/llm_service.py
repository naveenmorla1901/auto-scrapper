import json
from typing import Dict, Any, Optional, Tuple
import os
import re
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from ..models.schemas import CodeExecutionResult, TokenUsage
from ..utils.logger import app_logger, log_llm_interaction
from ..services.enhanced_llm import create_enhanced_scraping_prompt

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
            # According to DeepSeek API docs, the correct model name is:
            # - deepseek-chat
            "deepseek-coder": "deepseek-chat",
            "deepseek-chat": "deepseek-chat",
            "deepseek-llm-67b": "deepseek-chat",
            "deepseek-llm-7b": "deepseek-chat",

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
        app_logger.info(f"Attempting to set up coding LLM: {model}")
        self.coding_model_name = model # Store the user-provided name
        api_model_name = self.map_model_name(model) # Get potentially mapped name for API calls
        app_logger.info(f"Using API model name: {api_model_name}")

        provider = "unknown" # Initialize provider
        try:
            provider = self.get_provider(model)
            app_logger.info(f"Detected provider: {provider}")

            if not api_key:
                 raise ValueError(f"API key is missing for provider {provider}")

            # Common settings
            temperature = 0.1
            # Adjust max_tokens based on model/provider defaults if needed
            max_tokens = 4096 # Example: Set a sensible default max_tokens

            if provider == "openai":
                from langchain_openai import ChatOpenAI # Import within block
                self.coding_llm = ChatOpenAI(
                    model=api_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_key
                )
            elif provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI # Import within block
                self.coding_llm = ChatGoogleGenerativeAI(
                    model=api_model_name,
                    temperature=temperature,
                    google_api_key=api_key,
                    convert_system_message_to_human=True # May be needed
                    # Add safety_settings here if needed
                )
            elif provider == "anthropic":
                try:
                    from langchain_anthropic import ChatAnthropic # Import within block
                    self.coding_llm = ChatAnthropic(
                        model=api_model_name,
                        temperature=temperature,
                        max_tokens=max_tokens, # Anthropic uses max_tokens_to_sample or max_tokens
                        anthropic_api_key=api_key
                    )
                except ImportError:
                    error_msg = "Anthropic support requires langchain-anthropic package. Install it with: pip install langchain-anthropic"
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)
            elif provider == "mistral":
                 # ... (Mistral logic as before, ensuring imports are within try/except) ...
                 try:
                    from langchain_mistralai.chat_models import ChatMistralAI
                    self.coding_llm = ChatMistralAI(
                        model=api_model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        mistral_api_key=api_key
                    )
                 except ImportError:
                    try:
                        from langchain_community.chat_models import ChatMistralAI as ChatMistralAICommunity
                        app_logger.warning("langchain-mistralai not found, using langchain-community version.")
                        self.coding_llm = ChatMistralAICommunity(
                             model=api_model_name,
                             temperature=temperature,
                             max_tokens=max_tokens,
                             mistral_api_key=api_key
                         )
                    except ImportError:
                        error_msg = "Mistral AI support requires langchain-mistralai or langchain-community. Install with: pip install langchain-mistralai langchain-community"
                        app_logger.error(error_msg)
                        raise ImportError(error_msg)

            elif provider == "cohere":
                 try:
                    from langchain_cohere import ChatCohere # Import within block
                    self.coding_llm = ChatCohere(
                        model=api_model_name,
                        temperature=temperature,
                        cohere_api_key=api_key
                    )
                 except ImportError:
                    error_msg = "Cohere support requires langchain-cohere package. Install with: pip install langchain-cohere"
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)

            # --- Specific Provider Integrations from Community ---
            elif provider == "deepseek":
                try:
                    # First try direct DeepSeek integration using OpenAI SDK
                    try:
                        from openai import OpenAI
                        from langchain.chat_models.base import BaseChatModel
                        from langchain.schema import ChatResult, AIMessage, ChatGeneration
                        from typing import Optional

                        app_logger.info(f"Attempting direct DeepSeek integration using OpenAI SDK")

                        class DeepSeekChatModel(BaseChatModel):
                            """Custom chat model for DeepSeek API using OpenAI SDK."""

                            # Define class variables for Pydantic
                            model_name: str = "deepseek-chat"
                            api_key: str = None
                            temperature: float = 0.1
                            max_tokens: Optional[int] = 4096

                            def __init__(self, model_name, api_key, temperature=0.1, max_tokens=4096):
                                """Initialize with model name and API key."""
                                # Initialize parent class
                                super().__init__()
                                # Set instance variables
                                self.model_name = model_name
                                self.api_key = api_key
                                self.temperature = temperature
                                self.max_tokens = max_tokens
                                # Create OpenAI client with DeepSeek base URL
                                self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                                app_logger.info(f"Created DeepSeek client with model: {model_name}")

                            @property
                            def _llm_type(self) -> str:
                                """Return the type of LLM."""
                                return "deepseek_chat"

                            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                                """Generate a response from the DeepSeek API."""
                                # Convert LangChain messages to DeepSeek format
                                deepseek_messages = []
                                for message in messages:
                                    if isinstance(message, HumanMessage):
                                        deepseek_messages.append({"role": "user", "content": message.content})
                                    elif isinstance(message, SystemMessage):
                                        deepseek_messages.append({"role": "system", "content": message.content})
                                    else:
                                        deepseek_messages.append({"role": "assistant", "content": message.content})

                                # Prepare API parameters
                                params = {
                                    "model": self.model_name,
                                    "messages": deepseek_messages,
                                    "temperature": self.temperature,
                                    "max_tokens": min(self.max_tokens, 8192),  # DeepSeek limit is 8192
                                    "stream": False
                                }

                                # Add stop sequences if provided
                                if stop:
                                    params["stop"] = stop

                                try:
                                    # Make API call
                                    response = self._client.chat.completions.create(**params)
                                    content = response.choices[0].message.content

                                    # Create LangChain response format
                                    message = AIMessage(content=content)
                                    generation = ChatGeneration(message=message)
                                    return ChatResult(generations=[generation])
                                except Exception as e:
                                    app_logger.error(f"DeepSeek API error: {str(e)}")
                                    raise

                            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                                """Async version just calls the sync version for now."""
                                return self._generate(messages, stop, run_manager, **kwargs)

                        # Create the DeepSeek chat model
                        self.coding_llm = DeepSeekChatModel(
                            model_name="deepseek-chat",  # Use the standard model name for DeepSeek API
                            api_key=api_key,
                            temperature=temperature,
                            max_tokens=4096  # Safe value within DeepSeek's 8192 limit
                        )
                        app_logger.info("Successfully initialized direct DeepSeek integration")

                    except ImportError as ie:
                        app_logger.warning(f"Direct DeepSeek integration failed: {ie}. Trying TogetherAI fallback.")
                        raise
                    except Exception as e:
                        app_logger.warning(f"Direct DeepSeek integration failed: {e}. Trying TogetherAI fallback.")
                        raise

                except Exception:
                    # Fallback to TogetherAI
                    try:
                        # Use TogetherAI for DeepSeek models - with correct class name
                        from langchain_community.chat_models import ChatTogetherAI

                        app_logger.info(f"Attempting DeepSeek integration via TogetherAI")

                        # Format model name for Together API if needed
                        if not api_model_name.startswith("deepseek/"):
                            together_model_name = f"deepseek/{api_model_name}"
                            app_logger.info(f"Reformatting DeepSeek model name for Together API: {together_model_name}")
                        else:
                            together_model_name = api_model_name

                        self.coding_llm = ChatTogetherAI(
                            model=together_model_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            together_api_key=api_key
                        )
                        app_logger.info(f"Successfully initialized DeepSeek model via TogetherAI: {together_model_name}")
                    except ImportError:
                        error_msg = "DeepSeek integration requires either openai>=1.0.0 or langchain-community with ChatTogetherAI. Install with: pip install openai>=1.0.0 langchain-community>=0.0.10"
                        app_logger.error(error_msg)
                        raise ImportError(error_msg)
                    except Exception as e:
                        error_msg = f"Failed to initialize DeepSeek model: {e}"
                        app_logger.error(error_msg)
                        raise ValueError(error_msg)

            # --- Providers often accessed via Aggregators (like TogetherAI) ---
            # This block now handles providers intended to go through Together explicitly
            elif provider in ["together", "meta", "ai21"]: # Removed 'deepseek' from this list
                 try:
                    from langchain_community.chat_models import ChatTogether # Import within block
                    # Ensure api_key passed here is the TOGETHER_API_KEY
                    # Consider checking os.getenv("TOGETHER_API_KEY") as well
                    self.coding_llm = ChatTogether(
                        model=api_model_name, # Use the mapped name which might include provider prefix
                        temperature=temperature,
                        max_tokens=max_tokens,
                        together_api_key=api_key
                    )
                 except ImportError:
                    error_msg = f"{provider} support via Together AI requires langchain-community. Install with: pip install -U langchain-community"
                    app_logger.error(error_msg)
                    raise ImportError(error_msg)


            # --- Handle Ollama (Local Models) ---
            # elif provider == "ollama": # Example, uncomment and add pattern if needed
            #     try:
            #         from langchain_community.chat_models import ChatOllama
            #         ollama_model_name = api_model_name.split(':')[-1]
            #         self.coding_llm = ChatOllama(model=ollama_model_name, temperature=temperature)
            #     except ImportError:
            #         # ... error handling ...

            elif provider == "unknown":
                 raise ValueError(f"Provider for model '{model}' is unknown or unsupported by this configuration. Cannot set up LLM.")
            else:
                # This case implies a provider was detected but has no explicit handling block
                error_msg = f"Provider '{provider}' for model '{model}' is recognized but not configured in setup_coding_llm."
                app_logger.error(error_msg)
                raise ValueError(error_msg)

            app_logger.info(f"Successfully set up coding LLM: {model} (Provider: {provider}) using {type(self.coding_llm).__name__}")

        except ImportError as ie:
             # Log the specific import error
             app_logger.error(f"ImportError during setup for {provider} ('{model}'): {ie}. Please ensure required packages are installed.", exc_info=True)
             # Reraise a more informative error
             raise ImportError(f"Missing package for {provider}. Please install necessary libraries. Original error: {ie}") from ie
        except ValueError as ve: # Catch our specific ValueErrors (e.g., missing API key)
             app_logger.error(f"ValueError during setup for {provider} ('{model}'): {ve}", exc_info=True)
             self.coding_llm = None # Ensure LLM is not set
             raise ve # Reraise the ValueError
        except Exception as e:
            error_msg = f"An unexpected error occurred setting up coding LLM '{model}' (Provider: {provider}): {e}"
            app_logger.error(error_msg, exc_info=True) # Log stack trace
            self.coding_llm = None # Ensure LLM is not set if setup fails
            raise ValueError(error_msg) from e # Wrap and raise

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of a model call based on token usage"""
        # Get pricing for the model, or use default if not found
        pricing = self.pricing.get(model)
        if not pricing:
            app_logger.warning(f"Pricing not found for model: {model}. Using default pricing.")
            pricing = self.pricing.get("default", {"input": 0, "output": 0}) # Default to 0 if even default is missing

        # Calculate cost (convert from per 1000 tokens to per token)
        input_cost = (input_tokens / 1000) * pricing.get("input", 0)
        output_cost = (output_tokens / 1000) * pricing.get("output", 0)

        return input_cost + output_cost

    def update_token_usage(self, is_helper: bool, response: AIMessage) -> TokenUsage:
        """Update token usage for a model call using actual data from response metadata"""
        # Determine which usage to update
        usage_target = self.helper_token_usage if is_helper else self.coding_token_usage
        model = self.helper_model_name if is_helper else self.coding_model_name

        input_tokens = 0
        output_tokens = 0
        cost = 0.0

        # Extract token usage from response metadata if available
        # The structure might vary slightly between LangChain versions and providers
        if response and hasattr(response, 'response_metadata') and isinstance(response.response_metadata, dict):
            token_usage_data = response.response_metadata.get('token_usage', {})
            if isinstance(token_usage_data, dict): # Standard LangChain structure
                 input_tokens = token_usage_data.get('prompt_tokens', 0) or token_usage_data.get('input_tokens', 0)
                 output_tokens = token_usage_data.get('completion_tokens', 0) or token_usage_data.get('output_tokens', 0)
            else:
                 app_logger.warning(f"Unexpected token_usage format in response_metadata: {token_usage_data}")
        elif response and hasattr(response, 'usage_metadata') and isinstance(response.usage_metadata, dict):
             # Some newer versions might use usage_metadata
             input_tokens = response.usage_metadata.get('input_tokens', 0)
             output_tokens = response.usage_metadata.get('output_tokens', 0)
        else:
            app_logger.warning(f"Could not find token usage data in response metadata for model {model}. Falling back to estimating based on content length.")
            # Fallback estimation (less accurate) - keep it simple
            input_tokens = len(response.response_metadata.get('prompt', '')) // 4 if response and hasattr(response, 'response_metadata') else 0 # Rough estimate from prompt if possible
            output_tokens = len(response.content) // 4

        # Calculate cost based on actual tokens
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        # Update usage target
        usage_target.input_tokens += input_tokens
        usage_target.output_tokens += output_tokens
        usage_target.total_tokens = usage_target.input_tokens + usage_target.output_tokens
        usage_target.cost += cost

        app_logger.debug(f"Token update ({model}): Input={input_tokens}, Output={output_tokens}, Cost=${cost:.6f}")
        app_logger.debug(f"Total usage ({'Helper' if is_helper else 'Coding'}): Tokens={usage_target.total_tokens}, Cost=${usage_target.cost:.6f}")

        return usage_target

    def format_scraping_prompt(self, url: str, expected_data: str, website_analysis = None) -> str:
        """Use helper LLM to format the scraping prompt with optional website analysis"""
        if not self.helper_llm:
            self.setup_helper_llm()

        # Generate an example output format based on the expected data
        from ..services.scraper_helper import generate_example_output
        example_output = generate_example_output(expected_data)
        
        # Use the enhanced prompt creator
        prompt = create_enhanced_scraping_prompt(url, expected_data, example_output, website_analysis)

        app_logger.info(f"Formatting enhanced scraping prompt for URL: {url}")
        log_llm_interaction("INPUT", self.helper_model_name, prompt)

        # Track token usage
        start_time = time.time()

        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)

        # Update token usage using the actual response object
        self.update_token_usage(True, response)

        # Log execution time
        execution_time = time.time() - start_time
        app_logger.info(f"Helper LLM response time: {execution_time:.2f} seconds")

        log_llm_interaction("OUTPUT", self.helper_model_name, response=response.content)
        return response.content

    def extract_code_from_response(self, response) -> str:
        """Use helper LLM to extract clean code from the coding LLM's response

        Args:
            response: Either an AIMessage object or a string containing the LLM response

        Returns:
            str: The extracted code
        """
        if not self.helper_llm:
            self.setup_helper_llm()

        # Handle both AIMessage and string inputs
        response_content = response.content if hasattr(response, 'content') else response

        # First, try to extract code using simple regex for markdown code blocks
        import re
        python_code_blocks = re.findall(r'```python\s*(.*?)\s*```', response_content, re.DOTALL)
        if python_code_blocks:
            app_logger.info("Found Python code block using regex")
            return python_code_blocks[0].strip()

        # If no Python blocks, look for any code blocks
        code_blocks = re.findall(r'```\s*(.*?)\s*```', response_content, re.DOTALL)
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
        {response_content}
        """

        app_logger.info("Using helper LLM to extract code (regex methods failed)")
        log_llm_interaction("INPUT", self.helper_model_name, prompt)

        # Track token usage
        start_time = time.time()

        messages = [HumanMessage(content=prompt)]
        response = self.helper_llm.invoke(messages)

        # Update token usage using the actual response object
        self.update_token_usage(True, response)

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

    def generate_scraping_code(self, formatted_prompt: str) -> AIMessage:
        """Use coding LLM to generate scraping code. Returns the full AIMessage."""
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
        - Use Playwright or Selenium for dynamic sites that require JavaScript
        - Print the extracted data at the end as JSON
        - Design the JSON structure to best represent the specific data being extracted
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
        import re

        url = "https://example.com"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the data - structure will vary based on the specific data being extracted
        # This is just an example structure
        data = {
            "products": [
                {
                    "name": "Product 1",
                    "price": "$19.99",
                    "description": "Product description here"
                }
            ],
            "categories": ["Category 1", "Category 2"],
            "metadata": {
                "total_items": 42,
                "page": 1
            }
        }

        # Print the result as JSON
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

        # Add try-except block for robustness
        try:
            response = self.coding_llm.invoke(messages)
        except Exception as e:
            app_logger.error(f"Error invoking coding LLM ({self.coding_model_name}): {e}")
            # Re-raise or return a custom error object/message
            raise  # Re-raise the exception for now

        # Update token usage using the actual response object
        # Note: System message is part of the input tokens counted by the API
        self.update_token_usage(False, response)

        # Log execution time
        execution_time = time.time() - start_time
        app_logger.info(f"Coding LLM response time: {execution_time:.2f} seconds")

        log_llm_interaction("OUTPUT", self.coding_model_name, response=response.content)
        return response

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

        # Update token usage using the actual response object
        self.update_token_usage(True, response)

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

        # Update token usage using the actual response object
        self.update_token_usage(True, response)

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

        # Update token usage using the actual response object
        self.update_token_usage(True, response)

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

        # Update token usage using the actual response object
        self.update_token_usage(True, response)

        is_success = "YES" in response.content.upper()
        app_logger.info(f"Scraping success check result: {'SUCCESS' if is_success else 'FAILURE'}")
        log_llm_interaction("OUTPUT", self.helper_model_name, response=response.content)

        return is_success