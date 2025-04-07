"""
Website analyzer module for the Auto Web Scraper.
Provides comprehensive analysis of websites to improve scraping accuracy.
"""
import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin
from typing import TYPE_CHECKING, Dict, Any, List
from collections import defaultdict
import asyncio
import time
import traceback
import importlib.util

# Import dependency checker
from ..utils.dependency_checker import check_website_analyzer_dependencies
from ..utils.logger import app_logger

# Type checking imports
if TYPE_CHECKING:
    from ..models.schemas import WebsiteAnalysis

# Check if optional dependencies are available
LANGDETECT_AVAILABLE = importlib.util.find_spec("langdetect") is not None
NEST_ASYNCIO_AVAILABLE = importlib.util.find_spec("nest_asyncio") is not None

# Check if we're on Windows
import platform
IS_WINDOWS = platform.system() == "Windows"

# Playwright is not fully supported on Windows with asyncio
if IS_WINDOWS:
    PLAYWRIGHT_AVAILABLE = False
    app_logger.warning("Playwright dynamic analysis is disabled on Windows due to asyncio limitations")
else:
    PLAYWRIGHT_AVAILABLE = importlib.util.find_spec("playwright") is not None

# Import optional dependencies if available
if LANGDETECT_AVAILABLE:
    from langdetect import detect

# Log dependency status
app_logger.info(f"Website analyzer dependencies: Playwright: {PLAYWRIGHT_AVAILABLE}, LangDetect: {LANGDETECT_AVAILABLE}, Nest-AsyncIO: {NEST_ASYNCIO_AVAILABLE}")

async def analyze_website(url: str) -> 'WebsiteAnalysis':
    """
    Comprehensive webpage analyzer with parsing intelligence features.
    Returns detailed structure, content patterns, and scraping recommendations.

    Args:
        url: The URL of the website to analyze

    Returns:
        A WebsiteAnalysis object containing the analysis results
    """
    # Import here to avoid circular imports
    from ..models.schemas import WebsiteAnalysis
    start_time = time.time()
    app_logger.info(f"Starting website analysis for: {url}")

    report = {
        "url": url,
        "is_dynamic": False,
        # Structure Analysis
        "content_hierarchy": defaultdict(int),
        "components": {
            "navigation": 0,
            "forms": 0,
            "buttons": 0,
            "modals": 0,
            "tables": 0,
            "iframes": 0
        },
        "dynamic_attributes": [],
        # Content Analysis
        "text_ratio": 0,
        "language": "",
        "keyword_density": {},
        "structured_data": [],
        # Technical Analysis
        "frameworks": {
            "react": False,
            "angular": False,
            "vue": False
        },
        "apis": [],
        "pagination_patterns": [],
        "performance_metrics": {},
        # SEO & Security
        "seo_meta": {},
        "security_headers": {},
        "robots_txt": "",
        # Recommendations
        "selector_suggestions": [],
        "recommendations": [],
        "warnings": []
    }

    # --- Static Analysis ---
    try:
        app_logger.info(f"Performing static analysis for: {url}")

        # Fetch robots.txt
        try:
            robots_url = urljoin(url, "/robots.txt")
            robots_res = requests.get(robots_url, timeout=5)
            report["robots_txt"] = robots_res.text if robots_res.status_code == 200 else "Not found"
        except Exception as e:
            report["warnings"].append(f"Could not fetch robots.txt: {str(e)}")
            report["robots_txt"] = "Error fetching"

        # Fetch main page
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        static_response = requests.get(url, headers=headers, timeout=15)
        static_html = static_response.text
        soup = BeautifulSoup(static_html, "html.parser")

        # SEO Meta Analysis
        meta_tags = {}
        for tag in soup.find_all("meta"):
            name = tag.get("name") or tag.get("property") or tag.get("itemprop")
            if name:
                meta_tags[name] = tag.get("content")
        report["seo_meta"] = meta_tags

        # Security Headers
        security_headers = ["Content-Security-Policy", "X-Frame-Options", "Strict-Transport-Security"]
        report["security_headers"] = {h: static_response.headers.get(h) for h in security_headers}

        # Component Analysis
        report["components"].update({
            "navigation": len(soup.find_all(["nav", "header"])),
            "forms": len(soup.find_all("form")),
            "buttons": len(soup.find_all("button")),
            "modals": len(soup.find_all(class_=re.compile(r"modal|popup"))),
            "tables": len(soup.find_all("table")),
            "iframes": len(soup.find_all("iframe"))
        })

        # Framework Detection
        scripts = [script["src"].lower() for script in soup.find_all("script", src=True) if script.get("src")]
        report["frameworks"] = {
            "react": any("react" in script for script in scripts),
            "angular": any("angular" in script for script in scripts),
            "vue": any("vue" in script for script in scripts)
        }

        # Structured Data
        structured_data = []
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                if s.string:
                    structured_data.append(json.loads(s.string))
            except Exception:
                pass
        report["structured_data"] = structured_data

        # Content Hierarchy
        semantic_tags = ["article", "section", "header", "footer", "aside", "main"]
        for tag in semantic_tags:
            report["content_hierarchy"][tag] = len(soup.find_all(tag))
        report["content_hierarchy"]["content_containers"] = len(
            soup.find_all(class_=re.compile(r"content|container|main|body|wrapper"))
        )

        # Text Analysis
        text_length = len(soup.get_text(strip=True))
        html_length = len(static_html)
        report["text_ratio"] = round(text_length / html_length, 2) if html_length > 0 else 0

        # Language detection
        if LANGDETECT_AVAILABLE:
            try:
                text_sample = soup.get_text()[:1000]  # Use a sample to speed up detection
                if text_sample:
                    report["language"] = detect(text_sample)
            except Exception as e:
                report["warnings"].append(f"Language detection failed: {str(e)}")
        else:
            report["warnings"].append("Language detection not available (langdetect not installed)")

        # Keyword Density
        text = soup.get_text().lower()
        words = re.findall(r'\b\w{4,}\b', text)
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        report["keyword_density"] = dict(
            sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        )

        # Dynamic Attributes
        dynamic_attrs = set()
        for tag in soup.find_all(True):
            dynamic_attrs.update(
                attr for attr in tag.attrs if attr.startswith(('data-', 'aria-', 'ng-'))
            )
        report["dynamic_attributes"] = list(dynamic_attrs)

        # Pagination Patterns
        pagination_indicators = {
            "page_numbers": soup.find_all(string=re.compile(r'Page \d+')),
            "next_buttons": soup.find_all(href=re.compile(r'page|next')),
            "data_attributes": soup.find_all(attrs={"data-page": True})
        }
        report["pagination_patterns"] = [
            k for k, v in pagination_indicators.items() if v
        ]

        # Selector Suggestions
        id_candidates = [tag['id'] for tag in soup.find_all(id=True) if len(tag.text) > 100]
        if id_candidates:
            report["selector_suggestions"].append(
                f"High-content IDs: #{', #'.join(id_candidates[:5])}")  # Limit to 5 to avoid too much output

        # Look for main content containers
        content_containers = []
        for tag in soup.find_all(['article', 'main', 'div']):
            if tag.get('id') and re.search(r'content|main|article', tag.get('id')):
                content_containers.append(f"#{tag.get('id')}")
            elif tag.get('class'):
                for cls in tag.get('class'):
                    if re.search(r'content|main|article', cls):
                        content_containers.append(f".{cls}")

        if content_containers:
            report["selector_suggestions"].append(f"Main content containers: {', '.join(content_containers[:5])}")

        app_logger.info(f"Static analysis completed for: {url}")

    except Exception as e:
        error_details = traceback.format_exc()
        app_logger.error(f"Static analysis error: {str(e)}\n{error_details}")
        report["warnings"].append(f"Static analysis error: {str(e)}")

    # --- Dynamic Analysis ---
    try:
        app_logger.info(f"Starting dynamic analysis for: {url}")

        # Check if Playwright is available
        if PLAYWRIGHT_AVAILABLE:
            try:
                # Import Playwright only if it's available
                from playwright.async_api import async_playwright

                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()

                    # Set a reasonable timeout
                    await page.goto(url, timeout=20000)

                    # Gather performance metrics using modern Performance API
                    performance_metrics = await page.evaluate("""() => {
                        let loadTime = 0;
                        let domContentLoaded = 0;

                        // Use newer Performance API if available
                        const navEntries = performance.getEntriesByType('navigation');
                        if (navEntries && navEntries.length > 0) {
                            const nav = navEntries[0];
                            loadTime = nav.loadEventEnd;
                            domContentLoaded = nav.domContentLoadedEventEnd;
                        } else if (performance.timing) {
                            // Fallback to deprecated API
                            const timing = performance.timing;
                            loadTime = timing.loadEventEnd - timing.navigationStart;
                            domContentLoaded = timing.domContentLoadedEventEnd - timing.navigationStart;
                        }

                        return {
                            loadTime: loadTime,
                            domContentLoaded: domContentLoaded,
                            domSize: document.documentElement.outerHTML.length,
                            resourceCount: performance.getEntriesByType('resource').length
                        };
                    }""")

                    report["performance_metrics"] = performance_metrics

                    # Detect dynamic content
                    dynamic_html = await page.content()
                    report["is_dynamic"] = (len(static_html) != len(dynamic_html))

                    # Count elements after JS execution
                    element_counts = await page.evaluate("""() => {
                        return {
                            forms: document.querySelectorAll('form').length,
                            tables: document.querySelectorAll('table').length,
                            iframes: document.querySelectorAll('iframe').length,
                            dynamicElements: document.querySelectorAll('[data-*]').length
                        };
                    }""")

                    # Update component counts with dynamic elements
                    for key in ['forms', 'tables', 'iframes']:
                        if element_counts[key] > report["components"][key]:
                            report["components"][key] = element_counts[key]
                            report["is_dynamic"] = True

                    # API Monitoring
                    api_requests = []

                    def handle_request(request):
                        if request.resource_type in ("xhr", "fetch"):
                            api_info = {
                                "url": request.url,
                                "method": request.method
                            }
                            api_requests.append(api_info)

                    page.on("request", handle_request)

                    # Wait a bit to capture AJAX calls
                    await page.wait_for_timeout(3000)

                    # Scroll down to trigger lazy loading
                    await page.evaluate("""() => {
                        window.scrollTo(0, document.body.scrollHeight / 2);
                    }""")

                    await page.wait_for_timeout(2000)

                    report["apis"] = api_requests[:10]  # Limit to 10 to avoid too much output

                    await browser.close()

                    app_logger.info(f"Dynamic analysis completed for: {url}")
            except NotImplementedError:
                app_logger.warning("Playwright not supported on this system. Skipping dynamic analysis.")
                report["warnings"].append("Dynamic analysis skipped: Playwright not supported on this system")
                # Set some default values based on static analysis
                report["is_dynamic"] = False
                report["performance_metrics"] = {"loadTime": 0, "domContentLoaded": 0, "domSize": len(static_html), "resourceCount": 0}
            except Exception as e:
                error_details = traceback.format_exc()
                app_logger.error(f"Playwright initialization error: {str(e)}\n{error_details}")
                report["warnings"].append(f"Dynamic analysis skipped: {str(e)}")
                # Set some default values based on static analysis
                report["is_dynamic"] = False
                report["performance_metrics"] = {"loadTime": 0, "domContentLoaded": 0, "domSize": len(static_html), "resourceCount": 0}
        else:
            app_logger.warning("Playwright not installed. Skipping dynamic analysis.")
            report["warnings"].append("Dynamic analysis skipped: Playwright not installed")
            # Set some default values based on static analysis
            report["is_dynamic"] = False
            report["performance_metrics"] = {"loadTime": 0, "domContentLoaded": 0, "domSize": len(static_html), "resourceCount": 0}

    except Exception as e:
        error_details = traceback.format_exc()
        app_logger.error(f"Dynamic analysis error:\n{error_details}")
        report["warnings"].append(f"Dynamic analysis error: {str(e)}")

    # --- Generate Recommendations ---
    recs = []

    # Basic recommendations based on site characteristics
    if report["is_dynamic"]:
        recs.append("Use Playwright/Selenium for JavaScript rendering")
        if report["frameworks"]["react"] or report["frameworks"]["angular"] or report["frameworks"]["vue"]:
            framework_names = [name for name, detected in report["frameworks"].items() if detected]
            recs.append(f"{', '.join(framework_names).title()} framework detected: Wait for dynamic content to load")
    else:
        recs.append("Static content: Use Requests + BeautifulSoup for efficient scraping")

    if report["components"]["forms"] > 0:
        recs.append(f"Found {report['components']['forms']} forms: Consider form submission handling")

    if report["components"]["tables"] > 0:
        recs.append(f"Found {report['components']['tables']} tables: Use pandas for structured data extraction")

    if any(report["security_headers"].values()):
        recs.append("Security headers detected: Consider using proper headers in requests")

    if report["pagination_patterns"]:
        recs.append(f"Pagination detected: Implement pagination handling")

    # Selector recommendations
    if report["selector_suggestions"]:
        for suggestion in report["selector_suggestions"]:
            recs.append(f"Selector suggestion: {suggestion}")

    report["recommendations"] = recs

    # Convert defaultdict to regular dict for JSON serialization
    report["content_hierarchy"] = dict(report["content_hierarchy"])

    execution_time = time.time() - start_time
    app_logger.info(f"Website analysis completed in {execution_time:.2f} seconds")

    # Convert the dictionary to a WebsiteAnalysis object
    return WebsiteAnalysis(**report)

# Synchronous wrapper for the async function
def analyze_website_sync(url: str) -> 'WebsiteAnalysis':
    """
    Synchronous wrapper for the analyze_website function.

    Args:
        url: The URL of the website to analyze

    Returns:
        A WebsiteAnalysis object containing the analysis results
    """
    # Import here to avoid circular imports
    from ..models.schemas import WebsiteAnalysis
    # Apply nest_asyncio if available (for Jupyter/Colab compatibility)
    if NEST_ASYNCIO_AVAILABLE:
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except Exception as e:
            app_logger.warning(f"Failed to apply nest_asyncio: {str(e)}")

    return asyncio.run(analyze_website(url))
