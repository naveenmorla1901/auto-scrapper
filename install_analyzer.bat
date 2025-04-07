@echo off
echo Installing website analyzer dependencies...

:: Activate the virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
)

:: Install the required packages
pip install langdetect==1.0.9
pip install nest-asyncio==1.6.0
pip install playwright==1.42.0

:: Install Playwright browsers
echo Installing Playwright browsers...
python -m playwright install chromium

echo.
echo Website analyzer dependencies installed successfully!
echo.
echo You can now run the application with the website analyzer feature.
echo.
pause
