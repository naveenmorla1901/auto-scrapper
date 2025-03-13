# Auto Web Scraper Quick Start Guide

Follow these steps to get your Auto Web Scraper up and running:

## 1. Setup the Environment

### Clone the Repository

```bash
git clone https://github.com/yourusername/auto-web-scraper.git
cd auto-web-scraper
```

### Create a Virtual Environment

```bash
# For Linux/Mac
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Playwright Browsers

```bash
playwright install
```

## 2. Configure Environment Variables

Create a `.env` file in the project root:

```
# Helper LLM API key (your private key for Gemini 2.0 Flash-Lite)
GOOGLE_API_KEY=your_google_api_key_here

# Security settings
SECRET_KEY=your_random_secret_key_here

# App settings
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Timeout settings (in seconds)
DEFAULT_EXECUTION_TIMEOUT=60
MAX_ATTEMPTS=3
```

## 3. Run the Application

```bash
# For Linux/Mac
./run.sh

# For Windows
run.bat

# Or use directly
cd auto-scraper
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 4. Access the Web Interface

Open your browser and go to:
```
http://localhost:8000
```

## 5. Using the Scraper

1. Enter the URL of the website you want to scrape
2. Describe the data you want to extract
3. Select your preferred LLM model for code generation
4. Enter your API key for the selected model
5. Click "Generate & Run Scraper"
6. Wait for the system to process your request
7. View the extracted data and generated code

## 6. View Logs

Check the console and the `logs` directory for detailed logs showing each step of the process.

## Troubleshooting

- **Helper LLM Not Working**: Make sure your Google API key is valid and has access to Gemini 2.0 Flash-Lite
- **Module Not Found Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Scraping Fails**: Some websites have anti-scraping protection. The system tries to adapt, but might not always succeed
- **API Key Errors**: Verify you've entered the correct API key for your selected model