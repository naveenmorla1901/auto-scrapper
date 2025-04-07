# Auto Web Scraper

An intelligent web scraping tool that uses LLMs to generate and refine scraping code based on natural language descriptions.

## Features

- Accepts URL and description of desired data
- **Website Analyzer**: Automatically analyzes website structure and provides insights to improve scraping
- Uses Gemini 2.0 Flash-Lite as a helper LLM to format requests
- Uses user-provided LLM (GPT-4, Claude, etc.) to generate scraping code
- Automatically executes and tests the scraping code
- Refines the code if errors occur
- Presents extracted data and working code to the user
- Interactive UI with real-time status updates and token usage tracking
- User-friendly web interface

## Tech Stack

- **Backend**: FastAPI, Python 3.9+
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **LLM Integration**: LangChain
- **Web Scraping**: Playwright, BeautifulSoup, Requests
- **Website Analysis**: Playwright, BeautifulSoup, LangDetect
- **Containerization**: Docker

## Installation

### Prerequisites

- Python 3.9 or higher
- API keys for LLM providers (OpenAI, Google, Anthropic, etc.)
- Git

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/auto-web-scraper.git
   cd auto-web-scraper
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Website Analyzer Installation (Optional):

   The website analyzer feature requires additional dependencies. You can install them using the provided script:

   **Windows:**
   ```bash
   install_analyzer_windows.bat
   ```
   Note: On Windows, only static analysis is supported due to asyncio limitations with Playwright.

   **Linux/Mac:**
   ```bash
   install_analyzer.bat
   ```

   **Manual Installation:**
   ```bash
   pip install langdetect==1.0.9 nest-asyncio==1.6.0
   # For Linux/Mac only:
   pip install playwright==1.42.0
   python -m playwright install chromium
   ```

5. Install Playwright browsers:
   ```bash
   playwright install
   ```

6. Set up environment variables:
   - Create a `.env` file based on the `.env.example` template
   - Add your API keys for different LLM providers

7. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

7. Open your browser and navigate to http://localhost:8000

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t auto-web-scraper .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env auto-web-scraper
   ```

3. Open your browser and navigate to http://localhost:8000

## Usage

1. Enter the URL of the website you want to scrape
2. Describe the data you want to extract
3. Select your preferred LLM and provide your API key
4. Click "Generate & Run Scraper"
5. View the extracted data and the working code

## Security

- API keys are used only for requests and not stored in the backend
- Code execution happens in a secure, sandboxed environment
- User-provided code is validated and sanitized before execution

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.