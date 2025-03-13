import subprocess
import tempfile
import os
import time
import re
import ast
import sys
import importlib
import logging
# Wrap the resource import in try/except for Windows compatibility
try:
    import resource
except ImportError:
    resource = None
from collections import deque
from typing import Dict, List, Tuple, Optional

# Security settings
ALLOWED_PACKAGES = [
    "numpy", "pandas", "requests", 
    "beautifulsoup4", "selenium", "scrapy", 
    "lxml", "html5lib", "httpx", "parsel", 
    "fake-useragent", "playwright", "webdriver-manager",
    "selenium", "bs4", "urllib3"
]
MAX_PACKAGE_SIZE = 10  # MB
INSTALL_TIMEOUT = 15  # Seconds
DEFAULT_TIMEOUT = 60  # Seconds

# Scraping-specific error patterns
SCRAPING_ERROR_PATTERNS = {
    "Connection refused": "The website may be blocking automated requests",
    "HTTPError: 403": "Access forbidden - site may have anti-scraping measures",
    "HTTPError: 429": "Too many requests - rate limiting detected",
    "timeout": "Request timed out - site might be slow or blocking",
    "NoSuchElementException": "Element not found - selector may be incorrect or page structure changed",
    "ElementNotVisibleException": "Element not visible - may be hidden or not loaded yet",
    "captcha": "CAPTCHA detected - site is using anti-bot protection"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_scraping_errors(stderr: str) -> dict:
    """
    Analyze scraping-specific errors from stderr output
    and provide targeted recommendations
    """
    scraping_issues = []
    
    for pattern, explanation in SCRAPING_ERROR_PATTERNS.items():
        if pattern.lower() in stderr.lower():
            recommendation = ""
            
            # Add specific recommendations based on error type
            if "403" in pattern or "Connection refused" in pattern:
                recommendation = "Try adding request headers (User-Agent) or using a headless browser"
            elif "429" in pattern:
                recommendation = "Add delays between requests with time.sleep()"
            elif "timeout" in pattern:
                recommendation = "Increase request timeout or add retry mechanism"
            elif "NoSuchElementException" in pattern or "ElementNotVisibleException" in pattern:
                recommendation = "Check selector or add explicit waits for dynamic content"
            elif "captcha" in pattern:
                recommendation = "Site uses CAPTCHA protection - consider if scraping is appropriate"
            
            scraping_issues.append({
                "error_type": pattern,
                "explanation": explanation,
                "recommendation": recommendation
            })
    
    return scraping_issues



def safe_import(module_name: str) -> Optional[object]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None

# ==== ORIGINAL (unchanged) ====
black = safe_import('black')
autopep8 = safe_import('autopep8')
yapf = safe_import('yapf')
ruff = safe_import('ruff')
isort = safe_import('isort')

AVAILABLE_FORMATTERS = {
    'black': black is not None,
    'autopep8': autopep8 is not None,
    'yapf': yapf is not None,
    'ruff': ruff is not None,
    'isort': isort is not None,
}

def install_package(package: str) -> bool:
    """Secure package installer with constraints"""
    if package not in ALLOWED_PACKAGES:
        logger.error(f"Package {package} not allowed")
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", package],
            timeout=INSTALL_TIMEOUT,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Install failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Package installation timed out")
        return False

def detect_required_imports(code: str) -> List[str]:
    """AST-based import detection"""
    try:
        tree = ast.parse(code)
        imports = set()

        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            
            def visit_ImportFrom(self, node):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        ImportVisitor().visit(tree)
        return list(imports)
    except SyntaxError:
        return []

def secure_execution_environment():
    """Set resource limits and security constraints for the child process.
    This function runs only on Linux since chroot and resource limits are not supported on Windows."""
    if os.name == 'nt':
        logger.info("Secure execution environment not supported on Windows. Skipping security constraints.")
        return
    try:
        if resource:
            # Prevent fork bombs
            resource.setrlimit(resource.RLIMIT_NPROC, (50, 50))
            # Limit memory usage (50MB)
            resource.setrlimit(resource.RLIMIT_AS, (50 * 1024 * 1024, 50 * 1024 * 1024))
        # Attempt to restrict file system access via chroot
        temp_root = tempfile.mkdtemp()
        try:
            os.chroot(temp_root)
        except PermissionError as e:
            logger.error(f"chroot failed (requires root privileges): {e}")
        os.umask(0o777)
    except Exception as e:
        logger.error(f"Failed to set secure execution environment: {e}")

# ==== NEW: Critical preprocessing ====
def preprocess_code(code_string: str) -> str:
    """Standardize whitespace and basic formatting"""
    # Convert tabs to 4 spaces
    code = code_string.expandtabs(4)
    
    # Remove trailing whitespace while preserving empty lines
    lines = []
    for line in code.split('\n'):
        stripped = line.rstrip()
        if not stripped and line.strip() == '':
            lines.append('')  # Preserve empty lines
        else:
            lines.append(stripped)
    
    return '\n'.join(lines)

# ==== IMPROVED: Stack-based delimiter fixing ====
def fix_unclosed_delimiters(code_string: str) -> str:
    """Fix unclosed brackets/parens/braces at correct positions"""
    stack = deque()  # Each item: (char, line_num, col_num)
    lines = code_string.split('\n')
    
    for line_num, line in enumerate(lines):
        in_string = False
        string_char = None
        for col_num, char in enumerate(line):
            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                continue
            if not in_string:
                if char in '({[':
                    stack.append((char, line_num, col_num))
                elif char in ')}]':
                    if stack and ((char == ')' and stack[-1][0] == '(') or
                                  (char == '}' and stack[-1][0] == '{') or
                                  (char == ']' and stack[-1][0] == '[')):
                        stack.pop()

    # Add missing closers in reverse order
    while stack:
        opener, line_num, col_num = stack.pop()
        closer = {'(': ')', '{': '}', '[': ']'}[opener]
        original = lines[line_num]
        lines[line_num] = original[:col_num+1] + closer + original[col_num+1:]
        logger.warning(f"Added {closer} at line {line_num+1}, column {col_num+1}")

    return '\n'.join(lines)

# ==== IMPROVED: Recursive orphaned block fixing ====
def fix_orphaned_blocks(code_string: str, max_depth: int = 3) -> str:
    """Recursively fix orphaned control flow blocks"""
    def _fix_level(lines: List[str], depth: int) -> Tuple[List[str], bool]:
        fixed = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            if stripped.startswith(('else:', 'elif ', 'except:', 'finally:')):
                # Check for matching parent (if for else/elif, try for except/finally)
                parent_found = False
                for j in range(i-1, -1, -1):
                    prev_indent = len(lines[j]) - len(lines[j].lstrip())
                    if prev_indent < indent:
                        parent_found = any(
                            lines[j].strip().startswith(keyword)
                            for keyword in ['if ', 'try ']
                        )
                        break
                if not parent_found and depth < max_depth:
                    # Insert dummy parent (try: for except, else for others)
                    parent = 'try:' if 'except' in stripped else 'if True:'
                    lines.insert(i, ' ' * indent + parent)
                    lines.insert(i+1, ' ' * (indent + 4) + 'pass')
                    fixed = True
                    break
        return lines, fixed

    lines = code_string.split('\n')
    for depth in range(max_depth):
        lines, fixed = _fix_level(lines, depth)
        if not fixed:
            break
    return '\n'.join(lines)

# ==== NEW: Fix missing colons ====
def fix_missing_colons(code_string: str) -> str:
    """Add missing colons to common statement headers with more robust pattern matching."""
    lines = code_string.split('\n')
    keywords = ['def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
    new_lines = []
    
    # Using regular expressions to find statements without colons
    block_statement_pattern = re.compile(r'^(\s*)((?:' + '|'.join(keywords) + r')\b\s+[^:]*?)$')
    
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            new_lines.append(line)
            continue
        
        match = block_statement_pattern.match(line)
        if match:
            # We found a line that looks like a block statement without a colon
            indentation, statement = match.groups()
            
            # Check if there's a comment at the end
            comment_match = re.search(r'(#.*)$', statement)
            comment = comment_match.group(1) if comment_match else ""
            
            # Remove comment for the check
            statement_without_comment = re.sub(r'#.*$', '', statement).rstrip()
            
            # If it really doesn't end with a colon, add one
            if not statement_without_comment.endswith(':'):
                fixed_line = f"{indentation}{statement_without_comment}:{comment}"
                logger.info(f"Added missing colon on line {i+1}: '{line}' -> '{fixed_line}'")
                new_lines.append(fixed_line)
                continue
        
        # If we get here, either it's not a block statement or it already has a colon
        new_lines.append(line)
    
    return '\n'.join(new_lines)

# ==== NEW: Fix indentation in blocks ====
def fix_indentation_in_blocks(code_string: str) -> str:
    """Heuristically adjust indentation for blocks that follow headers ending with a colon."""
    lines = code_string.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        # If previous non-empty line ends with a colon, ensure current line is indented
        if i > 0:
            # Find previous non-empty line
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0 and lines[j].rstrip().endswith(':'):
                expected_indent = (len(lines[j]) - len(lines[j].lstrip())) + 4
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and current_indent < expected_indent:
                    line = ' ' * expected_indent + line.lstrip()
        new_lines.append(line)
    return '\n'.join(new_lines)

# ==== NEW: AST validation ====
def validate_syntax(code_string: str) -> bool:
    """Validate syntax of Python code string."""
    try:
        ast.parse(code_string)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}")
        return False

# Detailed version that returns error information
def validate_syntax_with_details(code_string: str) -> tuple[bool, str]:
    """Validate syntax and return details about any errors."""
    try:
        ast.parse(code_string)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}, column {e.offset}: {e.msg}"
# ==== IMPROVED: Execution flow ====
def execute_code(code_string: str, timeout: int = DEFAULT_TIMEOUT) -> Dict:
    """Secure yet fast execution with dynamic dependencies.
    Applies several syntax fixers before formatting and execution."""
    code_string = preprocess_code(code_string)
    fix_methods = []
    
    # Phase 1: Dependency handling
    required_imports = detect_required_imports(code_string)
    missing_imports = [imp for imp in required_imports if not safe_import(imp)]
    for package in missing_imports:
        if not install_package(package):
            return {
                "success": False,
                "stderr": f"Missing required package: {package}",
                "fix_methods": [],
                "scraping_issues": []
            }
    
    # Phase 2: Syntax fixing
    syntax_valid = validate_syntax(code_string)
    if not syntax_valid:
        logger.info("Attempting syntax fixes...")
        
        # Apply fix for missing colons first
        original_code = code_string
        code_string = fix_missing_colons(code_string)
        if code_string != original_code:
            fix_methods.append('colon_fix')
            logger.info("Applied colon fixes")
            
            # Check if that fixed the syntax
            syntax_valid = validate_syntax(code_string)
        
        # If still invalid, try fixing unclosed delimiters
        if not syntax_valid:
            original_code = code_string
            code_string = fix_unclosed_delimiters(code_string)
            if code_string != original_code:
                fix_methods.append('delimiter_fix')
                logger.info("Applied delimiter fixes")
                
                # Check if that fixed the syntax
                syntax_valid = validate_syntax(code_string)
        
        # If still invalid, try fixing orphaned blocks
        if not syntax_valid:
            original_code = code_string
            code_string = fix_orphaned_blocks(code_string)
            if code_string != original_code:
                fix_methods.append('block_fix')
                logger.info("Applied block fixes")
                
                # Check if that fixed the syntax
                syntax_valid = validate_syntax(code_string)
        
        # If still invalid, try fixing indentation
        if not syntax_valid:
            original_code = code_string
            code_string = fix_indentation_in_blocks(code_string)
            if code_string != original_code:
                fix_methods.append('indent_fix')
                logger.info("Applied indentation fixes")
                
                # Final syntax check
                syntax_valid = validate_syntax(code_string)
        
        # If we still have invalid syntax after all our fixes, give up
        if not syntax_valid:
            logger.warning("Unable to fix all syntax errors")
            return {
                "success": False,
                "stderr": "Unrecoverable syntax errors",
                "fix_methods": fix_methods,
                "scraping_issues": []
            }
    
    # Phase 3: Formatting (using formatter priority)
    formatting_applied = False
    formatter_priority = ['black', 'yapf', 'autopep8']  # Removed ruff since it's causing issues
    
    for formatter in formatter_priority:
        if AVAILABLE_FORMATTERS.get(formatter, False):
            try:
                formatter_func = globals()[f'format_with_{formatter}']
                formatted = formatter_func(code_string)
                
                # Only use formatted code if it's valid syntax
                if validate_syntax(formatted):
                    code_string = formatted
                    fix_methods.append(formatter)
                    formatting_applied = True
                    logger.info(f"Applied {formatter} formatting")
                    break
                else:
                    logger.warning(f"{formatter} produced invalid syntax, skipping")
            except Exception as e:
                logger.error(f"Error using {formatter}: {e}")
                # Continue to the next formatter
    
    if not formatting_applied:
        logger.warning("No formatters were successfully applied")
    
    # Detect what type of scraper we're dealing with
    scraper_type = detect_script_type(code_string)
    logger.info(f"Detected scraper type: {scraper_type}")
    
    # Phase 4: Secure execution in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "code.py")
        with open(file_path, "w") as f:
            f.write(code_string)
        
        try:
            start_time = time.time()
            # On Windows, preexec_fn is not supported
            preexec = secure_execution_environment if os.name != 'nt' else None
            process = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                shell=False,
                preexec_fn=preexec
            )
            return {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "success": True,
                "execution_time": time.time() - start_time,
                "fix_methods": fix_methods,
                "formatted_code": code_string,
                "scraper_type": scraper_type
            }
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr
            execution_time = time.time() - start_time
            
            # Analyze scraping-specific errors
            scraping_issues = analyze_scraping_errors(stderr_output)
            
            return {
                "stdout": e.stdout,
                "stderr": stderr_output,
                "success": False,
                "execution_time": execution_time,
                "fix_methods": fix_methods,
                "scraping_issues": scraping_issues,
                "scraper_type": scraper_type
            }
        except subprocess.TimeoutExpired:
            # Timeouts are common in web scraping - provide helpful feedback
            timeout_message = f"Execution timed out after {timeout}s"
            if scraper_type in ["selenium", "playwright"]:
                timeout_message += ". Browser-based scrapers often need longer timeouts or explicit waits."
            elif scraper_type == "requests":
                timeout_message += ". Consider adding timeout parameters to your requests."
            
            return {
                "stdout": "",
                "stderr": timeout_message,
                "success": False,
                "execution_time": timeout,
                "fix_methods": fix_methods,
                "scraping_issues": [
                    {
                        "error_type": "timeout",
                        "explanation": "The request or script execution took too long",
                        "recommendation": "Add explicit timeouts, reduce scope, or add pagination"
                    }
                ],
                "scraper_type": scraper_type
            }
        except Exception as e:
            # Catch any other exceptions that might occur
            return {
                "stdout": "",
                "stderr": f"Unexpected error: {str(e)}",
                "success": False,
                "execution_time": time.time() - start_time,
                "fix_methods": fix_methods,
                "scraping_issues": analyze_scraping_errors(str(e)),
                "scraper_type": scraper_type
            }
        
def detect_script_type(code_string: str) -> str:
    """
    Detect what type of web scraping the code is using
    to provide more specific error handling
    """
    if "selenium" in code_string or "webdriver" in code_string:
        return "selenium"
    elif "playwright" in code_string:
        return "playwright"
    elif "scrapy" in code_string:
        return "scrapy"
    elif "beautifulsoup" in code_string or "bs4" in code_string:
        return "beautifulsoup"
    elif "requests" in code_string:
        return "requests"
    else:
        return "unknown"
    
# ==== ORIGINAL FORMATTERS (modified with logging) ====
def format_with_black(code_string: str) -> str:
    if not AVAILABLE_FORMATTERS['black']:
        return code_string
    try:
        mode = black.Mode(line_length=88)
        return black.format_str(code_string, mode=mode)
    except Exception as e:
        logger.error(f"Black failed: {e}")
        return code_string

def format_with_ruff(code_string: str) -> str:
    """Format code using Ruff with better error handling"""
    if not AVAILABLE_FORMATTERS['ruff']:
        logger.warning("Ruff formatter not available")
        return code_string
    
    temp_file = None
    try:
        # Check if ruff is actually installed and accessible
        result = subprocess.run(['ruff', '--version'], 
                               capture_output=True, 
                               text=True, 
                               check=False)
        
        if result.returncode != 0:
            logger.error("Ruff binary not accessible in PATH")
            AVAILABLE_FORMATTERS['ruff'] = False
            return code_string
        
        # If we get here, ruff is available
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as f:
            temp_file = f.name
            f.write(code_string)
            f.flush()
        
        format_result = subprocess.run(['ruff', 'format', temp_file], 
                                      capture_output=True,
                                      text=True, 
                                      check=False)
        
        if format_result.returncode != 0:
            logger.error(f"Ruff formatting failed: {format_result.stderr}")
            return code_string
            
        with open(temp_file, 'r') as formatted:
            formatted_code = formatted.read()
        
        return formatted_code
    except Exception as e:
        logger.error(f"Ruff failed: {e}")
        return code_string
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error removing temp file: {e}")

def format_with_yapf(code_string: str) -> str:
    if not AVAILABLE_FORMATTERS['yapf']:
        return code_string
    try:
        from yapf.yapflib.yapf_api import FormatCode
        formatted_code, _ = FormatCode(code_string, style_config='pep8')
        return formatted_code
    except Exception as e:
        logger.error(f"YAPF failed: {e}")
        return code_string

def format_with_autopep8(code_string: str) -> str:
    if not AVAILABLE_FORMATTERS['autopep8']:
        return code_string
    try:
        return autopep8.fix_code(code_string, options={
            'aggressive': 2,
            'max_line_length': 88
        })
    except Exception as e:
        logger.error(f"autopep8 failed: {e}")
        return code_string
