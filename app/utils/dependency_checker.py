"""
Utility to check if optional dependencies are installed.
"""
import importlib.util
from ..utils.logger import app_logger

def is_module_installed(module_name: str) -> bool:
    """Check if a module is installed."""
    return importlib.util.find_spec(module_name) is not None

def check_website_analyzer_dependencies() -> bool:
    """Check if all dependencies for the website analyzer are installed."""
    import platform
    is_windows = platform.system() == "Windows"

    # On Windows, we don't need Playwright due to asyncio limitations
    if is_windows:
        required_modules = ["langdetect", "nest_asyncio"]
        install_script = "install_analyzer_windows.bat"
        app_logger.info("Running on Windows: Playwright dynamic analysis is disabled")
    else:
        required_modules = ["langdetect", "playwright", "nest_asyncio"]
        install_script = "install_analyzer.bat"

    missing_modules = []
    for module in required_modules:
        if not is_module_installed(module):
            missing_modules.append(module)

    if missing_modules:
        app_logger.warning(f"Website analyzer missing dependencies: {', '.join(missing_modules)}")
        app_logger.warning(f"Run '{install_script}' or install them manually to enable full functionality.")
        return False

    return True
