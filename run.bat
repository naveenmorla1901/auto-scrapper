@echo off
:: For Windows

:: Activate virtual environment
call venv\Scripts\activate

:: Run the FastAPI application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000