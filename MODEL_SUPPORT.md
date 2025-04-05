# LLM Model Support in Auto Web Scraper

Auto Web Scraper now supports a wide range of Large Language Models (LLMs) for generating web scraping code. This document provides details about the supported models and how to use them.

## Supported LLM Providers and Models

### OpenAI Models
- **GPT-4o**: Latest multimodal model with strong coding capabilities
- **GPT-4o Mini**: Smaller, faster version of GPT-4o
- **GPT-4 Turbo**: Enhanced version of GPT-4 with improved performance
- **GPT-4**: Strong general-purpose model with excellent coding abilities
- **GPT-3.5 Turbo**: Faster, more cost-effective option

### Google Models
- **Gemini 1.5 Pro**: Google's advanced multimodal model
- **Gemini 1.5 Flash**: Faster version of Gemini 1.5
- **Gemini 2.0 Flash-Lite**: Used as the helper LLM (server-side)
- **Gemini 1.0 Pro**: Previous generation model
- **Gemini 1.0 Ultra**: Previous generation premium model

### Anthropic Models
- **Claude 3 Opus**: Anthropic's most powerful model
- **Claude 3 Sonnet**: Balanced performance and speed
- **Claude 3 Haiku**: Fastest Claude model
- **Claude 3.5 Sonnet**: Latest Claude model with enhanced capabilities

### Mistral AI Models
- **Mistral Large**: Mistral's most powerful model
- **Mistral Medium**: Balanced performance model
- **Mistral Small**: Efficient, smaller model

### Meta Models (via Together AI)
- **Llama 3 70B**: Meta's largest open model
- **Llama 3 8B**: Smaller, efficient Llama model

### DeepSeek Models
- **DeepSeek Coder 33B**: Specialized for code generation
- **DeepSeek Chat 67B**: Large conversational model
- **DeepSeek LLM 67B**: Base large language model
- **DeepSeek LLM 7B**: Smaller base model

### Cohere Models
- **Command R**: Latest Cohere model
- **Command R Plus**: Enhanced version with more capabilities
- **Command Light**: Efficient, smaller model

### AI21 Models
- **Jurassic-2 Ultra**: AI21's most powerful model
- **Jurassic-2 Mid**: Mid-sized model with good performance

### Together AI Hosted Models
- **Yi 34B**: Yi large language model
- **Qwen 72B**: Alibaba's large language model
- **Falcon 180B**: One of the largest open models available

## API Keys

Each model requires its own API key from the respective provider:

- OpenAI models: [OpenAI API](https://platform.openai.com/)
- Google models: [Google AI Studio](https://makersuite.google.com/)
- Anthropic models: [Anthropic API](https://console.anthropic.com/)
- Mistral AI models: [Mistral AI Console](https://console.mistral.ai/)
- Together AI (for Meta, DeepSeek, and other models): [Together AI](https://www.together.ai/)
- Cohere models: [Cohere Platform](https://dashboard.cohere.com/)
- AI21 models: [AI21 Studio](https://studio.ai21.com/)

## Installation Requirements

To use these models, ensure you have the necessary packages installed:

```bash
pip install -r requirements.txt
```

The following packages are required for specific providers:
- OpenAI: `langchain-openai`
- Google: `langchain-google-genai`
- Anthropic: `langchain-anthropic`
- Mistral: `langchain-mistralai`
- Others (Together AI, Cohere, AI21): `langchain-community`

## Performance Considerations

Different models have different strengths when it comes to generating web scraping code:

- **Best for Complex Sites**: GPT-4o, Claude 3 Opus, Gemini 1.5 Pro
- **Best for Speed**: Claude 3 Haiku, Mistral Small, GPT-3.5 Turbo
- **Best for Code Quality**: DeepSeek Coder, GPT-4, Claude 3.5 Sonnet
- **Best Value**: Mistral Medium, Gemini 1.5 Flash, Claude 3 Sonnet

## Troubleshooting

If you encounter issues with a specific model:

1. Verify your API key is correct and has sufficient credits
2. Check that you have the required packages installed
3. Some models may have rate limits that affect performance
4. Try increasing the max_attempts parameter for more refinement cycles
