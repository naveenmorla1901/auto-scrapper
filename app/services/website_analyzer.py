"""
Enhanced website analyzer module for the Auto Web Scraper.
Provides comprehensive analysis of websites to improve scraping accuracy.
"""
import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import asyncio
import time
import traceback
import importlib.util
import os
import pickle
from pathlib import Path

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

# Cache directory for website analysis results
CACHE_DIR = Path("cache/website_analysis")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_selector(element) -> str:
    """Generate a CSS selector for an element"""
    if element.get('id'):
        return f"#{element.get('id')}"
    
    if element.get('class'):
        # Use the first class for simplicity
        if isinstance(element.get('class'), list) and element.get('class'):
            return f".{element.get('class')[0]}"
        elif isinstance(element.get('class'), str):
            return f".{element.get('class').split()[0]}"
    
    return element.name

def is_truly_dynamic(static_html: str, dynamic_html: str) -> bool:
    """
    Enhanced detection of dynamic content
    
    Args:
        static_html: HTML from static requests
        dynamic_html: HTML from browser rendering
        
    Returns:
        bool: True if the page is truly dynamic
    """
    # Simple length comparison as a first check
    if len(dynamic_html) > len(static_html) * 1.2:  # 20% size increase threshold
        return True
        
    # Compare text content (more reliable)
    static_soup = BeautifulSoup(static_html, "html.parser")
    dynamic_soup = BeautifulSoup(dynamic_html, "html.parser")
    
    static_text = static_soup.get_text(strip=True)
    dynamic_text = dynamic_soup.get_text(strip=True)
    
    if len(dynamic_text) > len(static_text) * 1.2:  # 20% text increase threshold
        return True
        
    # Compare important content containers
    static_containers = static_soup.find_all(class_=re.compile(r"content|container|results|list|feed|items|products|grid"))
    dynamic_containers = dynamic_soup.find_all(class_=re.compile(r"content|container|results|list|feed|items|products|grid"))
    
    if len(dynamic_containers) > len(static_containers) * 1.2:
        return True
    
    # Check for newly added content elements (dynamic content markers)
    static_items = len(static_soup.find_all(["article", "div", "li"], class_=re.compile(r"item|card|product|post|entry|result")))
    dynamic_items = len(dynamic_soup.find_all(["article", "div", "li"], class_=re.compile(r"item|card|product|post|entry|result")))
    
    if dynamic_items > static_items * 1.1:  # 10% more items
        return True
    
    return False

async def detect_interactive_elements(page) -> List[Dict[str, str]]:
    """
    Find elements that likely trigger content loading
    
    Args:
        page: Playwright page object
        
    Returns:
        List of interactive elements
    """
    return await page.evaluate("""() => {
        const clickables = [];
        
        // Buttons that might load content
        const buttonPatterns = /load|view|show|more|next|continue|expand|pagination/i;
        document.querySelectorAll('button, [role="button"], .btn, a[href="#"], [class*="load"], [class*="more"]').forEach(el => {
            if (el.textContent && el.textContent.match(buttonPatterns)) {
                clickables.push({
                    selector: getUniqueSelector(el),
                    text: el.textContent.trim().substring(0, 30),
                    type: 'button'
                });
            }
        });
        
        // Pagination elements
        document.querySelectorAll('.pagination a, [class*="pag"] a, [class*="page"] a, [aria-label*="page"]').forEach(el => {
            clickables.push({
                selector: getUniqueSelector(el),
                text: el.textContent.trim().substring(0, 30),
                type: 'pagination'
            });
        });
        
        // Custom selector generator
        function getUniqueSelector(el) {
            if (el.id) return `#${el.id}`;
            
            // Try class selector
            if (el.className && typeof el.className === 'string' && el.className.trim()) {
                const classes = el.className.trim().split(/\\s+/);
                for (const cls of classes) {
                    const selector = `.${cls}`;
                    if (document.querySelectorAll(selector).length < 5) {
                        return selector;
                    }
                }
            }
            
            // Try parent-child selector
            const parent = el.parentElement;
            if (parent && parent.id) {
                const children = Array.from(parent.children);
                const index = children.indexOf(el);
                if (index !== -1) {
                    return `#${parent.id} > ${el.tagName.toLowerCase()}:nth-child(${index + 1})`;
                }
            }
            
            // Fallback to basic tag
            return el.tagName.toLowerCase();
        }
        
        return clickables;
    }""")

def analyze_content_sequence(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Identify repeating patterns in content structure
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        List of content patterns
    """
    content_patterns = []
    
    # Find elements that have multiple children with similar structure
    for parent in soup.find_all(["div", "main", "section", "ul", "ol", "table"]):
        children = parent.find_all(recursive=False)
        
        # Need at least 3 child elements to detect a pattern
        if len(children) >= 3:
            # Check if children have similar structure
            first_three_children = children[:3]
            
            # Check if they have the same tag
            if len(set(child.name for child in first_three_children)) == 1:
                tag_name = first_three_children[0].name
                
                # Check if they have similar classes
                class_similarity = True
                if all(child.get('class') for child in first_three_children):
                    class_sets = [set(child.get('class')) for child in first_three_children]
                    if not (class_sets[0] & class_sets[1] & class_sets[2]):  # No common classes
                        class_similarity = False
                
                # Check if they have similar number of children
                child_counts = [len(child.find_all()) for child in first_three_children]
                count_similarity = (max(child_counts) - min(child_counts)) <= 2
                
                if class_similarity and count_similarity:
                    # This looks like a content pattern
                    pattern = {
                        'parent_selector': get_selector(parent),
                        'child_selector': f"{get_selector(parent)} > {tag_name}",
                        'count': len(children),
                        'confidence': 'high' if all(child.get('class') for child in first_three_children) else 'medium'
                    }
                    content_patterns.append(pattern)
    
    # Special case for item listings
    list_items = soup.find_all('li')
    if len(list_items) >= 5:  # At least 5 list items
        # Group by parent
        by_parent = defaultdict(list)
        for li in list_items:
            if li.parent:
                by_parent[li.parent].append(li)
        
        # Find parents with multiple children
        for parent, items in by_parent.items():
            if len(items) >= 5:  # Parent with at least 5 list items
                pattern = {
                    'parent_selector': get_selector(parent),
                    'child_selector': f"{get_selector(parent)} > li",
                    'count': len(items),
                    'confidence': 'high'
                }
                content_patterns.append(pattern)
    
    return content_patterns

def visualize_dom_structure(soup: BeautifulSoup, max_depth: int = 3) -> List[str]:
    """
    Generate a visual representation of the DOM hierarchy
    
    Args:
        soup: BeautifulSoup object
        max_depth: Maximum depth to visualize
        
    Returns:
        List of strings representing the DOM structure
    """
    structure = []
    
    def process_element(element, depth=0):
        if depth > max_depth:
            return
            
        # Get element info
        tag = element.name
        class_attr = element.get('class')
        class_str = f".{'.'.join(class_attr)}" if class_attr and isinstance(class_attr, list) else ""
        id_attr = element.get('id')
        id_str = f"#{id_attr}" if id_attr else ""
        
        # Create element representation
        element_str = f"{' ' * depth}|_ {tag}{id_str}{class_str}"
        
        # Add to structure
        structure.append(element_str)
        
        # Process children (limited to avoid huge outputs)
        for i, child in enumerate(element.find_all(recursive=False)):
            if i >= 10:  # Limit to 10 children per element
                structure.append(f"{' ' * (depth+1)}|_ ... ({len(element.find_all(recursive=False)) - 10} more)")
                break
            process_element(child, depth + 1)
    
    # Start with body tag
    body = soup.find('body')
    if body:
        process_element(body)
        
    return structure

def generate_reliable_selectors(soup: BeautifulSoup, target_elements: List) -> List[Dict[str, Any]]:
    """
    Generate multiple selector strategies with reliability scores
    
    Args:
        soup: BeautifulSoup object
        target_elements: List of elements to generate selectors for
        
    Returns:
        List of selector strategies
    """
    selector_strategies = []
    
    for element in target_elements:
        strategies = []
        
        # Strategy 1: ID-based (most reliable)
        if element.get('id'):
            strategies.append({
                'selector': f"#{element.get('id')}",
                'reliability': 0.9,
                'method': 'id'
            })
        
        # Strategy 2: Class-based
        if element.get('class'):
            class_selector = f".{'.'.join(element.get('class'))}" if isinstance(element.get('class'), list) else f".{element.get('class')}"
            
            # Check uniqueness
            count = len(soup.select(class_selector))
            reliability = 0.8 if count < 5 else (0.6 if count < 20 else 0.4)
            
            strategies.append({
                'selector': class_selector,
                'reliability': reliability,
                'method': 'class',
                'count': count
            })
        
        # Strategy 3: Parent-child structure
        parent = element.parent
        if parent:
            if parent.get('id'):
                # Child position
                siblings = parent.find_all(element.name, recursive=False)
                position = None
                for i, sibling in enumerate(siblings):
                    if sibling is element:
                        position = i + 1
                        break
                
                if position:
                    strategies.append({
                        'selector': f"#{parent.get('id')} > {element.name}:nth-child({position})",
                        'reliability': 0.8,
                        'method': 'parent-child-id'
                    })
            elif parent.get('class'):
                # Less reliable but still good
                parent_selector = f".{'.'.join(parent.get('class'))}" if isinstance(parent.get('class'), list) else f".{parent.get('class')}"
                
                # Child position
                siblings = parent.find_all(element.name, recursive=False)
                position = None
                for i, sibling in enumerate(siblings):
                    if sibling is element:
                        position = i + 1
                        break
                
                if position:
                    selector = f"{parent_selector} > {element.name}:nth-child({position})"
                    strategies.append({
                        'selector': selector,
                        'reliability': 0.7,
                        'method': 'parent-child-class'
                    })
            
        # Strategy 4: Text content anchor (for unique text)
        if element.string and len(element.string.strip()) > 10:
            text_content = element.string.strip()[:20]
            # Need to use an XPath-like selector for text content in CSS
            strategies.append({
                'selector': f"{element.name}:contains('{text_content}')",
                'reliability': 0.5,
                'method': 'text-content',
                'note': 'Requires jQuery or XPath'
            })
        
        # Strategy 5: Attribute-based
        for attr_name, attr_value in element.attrs.items():
            if attr_name not in ['class', 'id', 'style']:
                if isinstance(attr_value, str) and len(attr_value) < 50:  # Reasonable length
                    strategies.append({
                        'selector': f"{element.name}[{attr_name}='{attr_value}']",
                        'reliability': 0.7,
                        'method': 'attribute'
                    })
        
        # Add the best strategies
        if strategies:
            # Sort by reliability
            strategies.sort(key=lambda x: x['reliability'], reverse=True)
            selector_strategies.append({
                'element': element.name,
                'strategies': strategies[:3]  # Top 3 strategies
            })
    
    return selector_strategies

async def test_extraction_heuristics(soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
    """
    Try common extraction patterns to identify what works for this site
    
    Args:
        soup: BeautifulSoup object
        url: URL of the website
        
    Returns:
        List of extraction results
    """
    extraction_results = []
    
    # Test pattern 1: List items
    list_items = soup.find_all('li')
    if len(list_items) > 5:
        sample = [item.text.strip()[:50] for item in list_items[:5] if item.text.strip()]
        extraction_results.append({
            'pattern': 'list_items',
            'selector': 'li',
            'sample': sample,
            'count': len(list_items)
        })
    
    # Test pattern 2: Card-like elements
    cards = soup.find_all(class_=re.compile(r'card|item|box|tile|product'))
    if len(cards) > 3:
        sample = [card.text.strip()[:50] for card in cards[:3] if card.text.strip()]
        extraction_results.append({
            'pattern': 'cards',
            'selector': '[class*="card"], [class*="item"], [class*="box"], [class*="tile"], [class*="product"]',
            'sample': sample,
            'count': len(cards)
        })
    
    # Test pattern 3: Table rows
    rows = soup.find_all('tr')
    if len(rows) > 3:
        sample = [row.text.strip()[:50] for row in rows[:3] if row.text.strip()]
        extraction_results.append({
            'pattern': 'table_rows',
            'selector': 'tr',
            'sample': sample,
            'count': len(rows)
        })
    
    # Test pattern 4: Article/post elements
    articles = soup.find_all(['article', 'div'], class_=re.compile(r'post|article|story|entry|feed-item'))
    if len(articles) > 2:
        sample = [article.text.strip()[:50] for article in articles[:3] if article.text.strip()]
        extraction_results.append({
            'pattern': 'articles',
            'selector': 'article, [class*="post"], [class*="article"], [class*="story"], [class*="entry"], [class*="feed-item"]',
            'sample': sample,
            'count': len(articles)
        })
    
    # Test pattern 5: Heading elements with content
    headings = soup.find_all(['h1', 'h2', 'h3'])
    if len(headings) > 3:
        sample = [h.text.strip()[:50] for h in headings[:5] if h.text.strip()]
        extraction_results.append({
            'pattern': 'headings',
            'selector': 'h1, h2, h3',
            'sample': sample,
            'count': len(headings)
        })
    
    # Sort by count (higher first)
    extraction_results.sort(key=lambda x: x['count'], reverse=True)
    
    return extraction_results

async def capture_visual_analysis(page) -> Dict[str, Any]:
    """
    Capture screenshots and analyze visual structure
    
    Args:
        page: Playwright page object
        
    Returns:
        Visual analysis data
    """
    visual_data = {}
    
    # Create a directory for screenshots
    screenshots_dir = Path("static/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique ID for this analysis
    analysis_id = int(time.time())
    
    # Take full page screenshot
    screenshot_path = f"static/screenshots/full_page_{analysis_id}.png"
    await page.screenshot(path=screenshot_path, full_page=True)
    visual_data['full_screenshot'] = screenshot_path
    
    # Identify visually distinct sections
    sections = await page.evaluate("""() => {
        // Get all visually prominent containers
        const sections = [];
        document.querySelectorAll('div, section, article, main, aside').forEach(el => {
            const rect = el.getBoundingClientRect();
            // Only consider visible and reasonably sized elements
            if (rect.width > 200 && rect.height > 200 && 
                rect.top < window.innerHeight && 
                rect.left < window.innerWidth) {
                
                const styles = window.getComputedStyle(el);
                
                // Check if it has visual distinction
                if (styles.backgroundColor !== 'rgba(0, 0, 0, 0)' || 
                    styles.border !== 'none' ||
                    styles.boxShadow !== 'none' ||
                    styles.borderRadius !== '0px') {
                    
                    // Get a unique selector
                    let selector = '';
                    if (el.id) {
                        selector = `#${el.id}`;
                    } else if (el.className) {
                        const classes = el.className.split(' ').filter(c => c);
                        if (classes.length > 0) {
                            selector = `.${classes[0]}`;
                        } else {
                            selector = el.tagName.toLowerCase();
                        }
                    } else {
                        selector = el.tagName.toLowerCase();
                    }
                    
                    sections.push({
                        selector: selector,
                        width: rect.width,
                        height: rect.height,
                        x: rect.x,
                        y: rect.y,
                        visible: true
                    });
                }
            }
        });
        return sections;
    }""")
    
    # Take screenshots of each section
    section_screenshots = []
    for i, section in enumerate(sections):
        if i >= 5:  # Limit to 5 sections
            break
            
        section_path = f"static/screenshots/section_{analysis_id}_{i}.png"
        try:
            await page.screenshot(
                path=section_path,
                clip={"x": section["x"], "y": section["y"], 
                      "width": section["width"], "height": section["height"]}
            )
            section_screenshots.append({
                'selector': section['selector'],
                'path': section_path,
                'dimensions': f"{section['width']}x{section['height']}"
            })
        except Exception as e:
            app_logger.warning(f"Failed to capture section screenshot: {e}")
    
    visual_data['sections'] = section_screenshots
    
    return visual_data

def select_code_template(website_analysis) -> str:
    """
    Select the most appropriate code template based on website analysis
    
    Args:
        website_analysis: WebsiteAnalysis object
        
    Returns:
        Code template string
    """
    templates = {
        'e-commerce': """
            # E-commerce scraping template with product extraction
            import requests
            from bs4 import BeautifulSoup
            import json
            
            url = "{url}"
            headers = {{
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }}
            
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # Raise exception for HTTP errors
                soup = BeautifulSoup(response.content, 'html.parser')
                
                products = []
                # Look for product containers
                product_elements = soup.select("{product_selector}")
                
                for element in product_elements:
                    product = {{}}
                    
                    # Extract product details
                    name_element = element.select_one("{name_selector}")
                    if name_element:
                        product["name"] = name_element.text.strip()
                    
                    price_element = element.select_one("{price_selector}")
                    if price_element:
                        product["price"] = price_element.text.strip()
                    
                    # Add product if it has at least a name
                    if "name" in product:
                        products.append(product)
                    
                # Print results as JSON
                print(json.dumps({{"products": products}}, indent=2))
                
            except requests.exceptions.RequestException as e:
                print(json.dumps({{"error": str(e)}}))
            except Exception as e:
                print(json.dumps({{"error": str(e)}}))
        """,
        
        'news': """
            # News site scraping template
            import requests
            from bs4 import BeautifulSoup
            import json
            
            url = "{url}"
            headers = {{
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }}
            
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # Raise exception for HTTP errors
                soup = BeautifulSoup(response.content, 'html.parser')
                
                articles = []
                # Look for article containers
                article_elements = soup.select("{article_selector}")
                
                for element in article_elements:
                    article = {{}}
                    
                    # Extract article details
                    title_element = element.select_one("{title_selector}")
                    if title_element:
                        article["title"] = title_element.text.strip()
                    
                    # Add article if it has a title
                    if "title" in article:
                        articles.append(article)
                    
                # Print results as JSON
                print(json.dumps({{"articles": articles}}, indent=2))
                
            except requests.exceptions.RequestException as e:
                print(json.dumps({{"error": str(e)}}))
            except Exception as e:
                print(json.dumps({{"error": str(e)}}))
        """,
        
        'dynamic_content': """
            # Dynamic content scraping with Playwright
            from playwright.sync_api import sync_playwright
            import json
            
            def scrape():
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    page = browser.new_page()
                    page.goto("{url}")
                    
                    # Wait for content to load
                    page.wait_for_selector("{content_selector}")
                    
                    # Extract items
                    items = []
                    elements = page.query_selector_all("{item_selector}")
                    
                    for element in elements:
                        text = element.text_content()
                        if text.strip():
                            items.append({{"content": text.strip()}})
                    
                    browser.close()
                    return items
            
            try:
                results = scrape()
                print(json.dumps({{"items": results}}, indent=2))
            except Exception as e:
                print(json.dumps({{"error": str(e)}}))
        """
    }
    
    if website_analysis.is_dynamic:
        return templates["dynamic_content"]
    
    # Check for e-commerce indicators
    ecommerce_terms = ['product', 'price', 'shop', 'cart', 'store', 'buy']
    if any(term in k.lower() for k in website_analysis.keyword_density for term in ecommerce_terms):
        return templates["e-commerce"]
    
    # Check for news indicators
    news_terms = ['article', 'news', 'post', 'blog', 'story']
    if any(term in k.lower() for k in website_analysis.keyword_density for term in news_terms):
        return templates["news"]
    
    # Default to dynamic content for safety
    return templates["dynamic_content"]

def get_cached_analysis(url: str) -> Optional[Dict[str, Any]]:
    """
    Get cached website analysis if available
    
    Args:
        url: URL of the website
        
    Returns:
        Cached analysis or None
    """
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{url_hash}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                
            # Check if cache is fresh (less than 24 hours old)
            cache_time = cached_data.get("cache_time", 0)
            if time.time() - cache_time < 86400:  # 24 hours
                app_logger.info(f"Using cached analysis for: {url}")
                return cached_data.get("analysis")
        except Exception as e:
            app_logger.warning(f"Failed to load cached analysis: {e}")
    
    return None

def cache_analysis(url: str, analysis: Dict[str, Any]) -> None:
    """
    Cache website analysis for future use
    
    Args:
        url: URL of the website
        analysis: Analysis data to cache
    """
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{url_hash}.pkl"
    
    try:
        with open(cache_file, "wb") as f:
            pickle.dump({
                "analysis": analysis,
                "cache_time": time.time()
            }, f)
        app_logger.info(f"Cached analysis for: {url}")
    except Exception as e:
        app_logger.warning(f"Failed to cache analysis: {e}")

async def enhanced_website_analysis(url: str) -> 'WebsiteAnalysis':
    """
    Enhanced two-phase website analysis
    
    Args:
        url: URL of the website
        
    Returns:
        WebsiteAnalysis object
    """
    # Import here to avoid circular imports
    from ..models.schemas import WebsiteAnalysis
    
    # Check for cached analysis
    cached_analysis = get_cached_analysis(url)
    if cached_analysis:
        # Convert dict to WebsiteAnalysis object
        return WebsiteAnalysis(**cached_analysis)
    
    start_time = time.time()
    app_logger.info(f"Starting enhanced website analysis for: {url}")
    
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
        # Enhanced Analysis
        "content_patterns": [],
        "extraction_test_results": [],
        "interactive_elements": [],
        "visual_structure": {},
        "selector_strategies": [],
        "dom_visualization": [],
        "code_templates": {},
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
        
        # Enhanced Analysis: Content Patterns
        report["content_patterns"] = analyze_content_sequence(soup)
        
        # Enhanced Analysis: Test Extraction Heuristics
        extraction_results = await test_extraction_heuristics(soup, url)
        report["extraction_test_results"] = extraction_results
        
        # Enhanced Analysis: DOM Visualization
        report["dom_visualization"] = visualize_dom_structure(soup)
        
        # Enhanced Analysis: Selector Strategies
        # Find potential content containers
        content_containers = soup.find_all(["div", "main", "section", "article"], 
                                          class_=re.compile(r"content|container|main|list|results"))[:5]
        report["selector_strategies"] = generate_reliable_selectors(soup, content_containers)
        
        # Selector Suggestions (original implementation)
        id_candidates = [tag['id'] for tag in soup.find_all(id=True) if len(tag.text) > 100]
        if id_candidates:
            report["selector_suggestions"].append(
                f"High-content IDs: #{', #'.join(id_candidates[:5])}")  # Limit to 5
        
        # Look for main content containers
        content_containers = []
        for tag in soup.find_all(['article', 'main', 'div']):
            if tag.get('id') and re.search(r'content|main|article', tag.get('id')):
                content_containers.append(f"#{tag.get('id')}")
            elif tag.get('class'):
                for cls in tag.get('class'):
                    if isinstance(cls, str) and re.search(r'content|main|article', cls):
                        content_containers.append(f".{cls}")
        
        if content_containers:
            report["selector_suggestions"].append(f"Main content containers: {', '.join(content_containers[:5])}")
        
        # Enhanced Analysis: Code Templates
        # Store selectors to use in templates
        content_selectors = {}
        
        # Container selectors
        containers = soup.find_all(["div", "main", "section"], class_=re.compile(r"content|container|main"))
        if containers:
            content_selectors["container"] = get_selector(containers[0])
        
        # Item selectors
        items = soup.find_all(["article", "div", "li"], class_=re.compile(r"item|card|product"))
        if items:
            content_selectors["item"] = get_selector(items[0])
        
        # Title selectors
        titles = soup.find_all(["h1", "h2", "h3", "a"], class_=re.compile(r"title|heading"))
        if titles:
            content_selectors["title"] = get_selector(titles[0])
        
        # Price selectors (for e-commerce)
        prices = soup.find_all(["span", "div"], class_=re.compile(r"price"))
        if prices:
            content_selectors["price"] = get_selector(prices[0])
        
        report["code_templates"] = {"selectors": content_selectors}
        
        app_logger.info(f"Static analysis completed for: {url}")
        
    except Exception as e:
        error_details = traceback.format_exc()
        app_logger.error(f"Static analysis error: {str(e)}\n{error_details}")
        report["warnings"].append(f"Static analysis error: {str(e)}")
    
    # Initial check to see if the site might be dynamic
    likely_dynamic = (
        report["frameworks"]["react"] or 
        report["frameworks"]["angular"] or 
        report["frameworks"]["vue"] or 
        len(report["dynamic_attributes"]) > 5 or
        report["text_ratio"] < 0.1  # Very low text ratio often indicates dynamic content
    )
    
    # --- Dynamic Analysis ---
    if PLAYWRIGHT_AVAILABLE:
        try:
            app_logger.info(f"Starting dynamic analysis for: {url}")
            
            # Import Playwright only if it's available
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set a user agent
                await page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                })
                
                # Set a reasonable timeout
                try:
                    await page.goto(url, timeout=30000)
                except Exception as e:
                    app_logger.warning(f"Page navigation timeout or error: {e}")
                    report["warnings"].append(f"Navigation timeout or error: {e}")
                
                # Wait for content to load
                try:
                    # Wait for common content containers
                    selectors = [
                        "main", "article", "#content", "#main", ".content", ".main",
                        "[role='main']", ".container", "#container"
                    ]
                    selector_str = ", ".join(selectors)
                    await page.wait_for_selector(selector_str, timeout=5000, state="attached")
                except Exception:
                    # If no specific container is found, just wait a bit
                    await page.wait_for_timeout(5000)
                
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
                        resourceCount: performance.getEntriesByType('resource').length,
                        ajaxCount: performance.getEntriesByType('resource')
                            .filter(r => r.initiatorType === 'xmlhttprequest').length
                    };
                }""")
                
                report["performance_metrics"] = performance_metrics
                
                # Detect dynamic content
                dynamic_html = await page.content()
                report["is_dynamic"] = is_truly_dynamic(static_html, dynamic_html)
                
                # Count elements after JS execution
                element_counts = await page.evaluate("""() => {
                    return {
                        forms: document.querySelectorAll('form').length,
                        tables: document.querySelectorAll('table').length,
                        iframes: document.querySelectorAll('iframe').length,
                        dynamicElements: document.querySelectorAll('[data-*]').length,
                        buttons: document.querySelectorAll('button, [role="button"], .btn').length,
                        links: document.querySelectorAll('a').length
                    };
                }""")
                
                # Update component counts with dynamic elements
                for key in ['forms', 'tables', 'iframes', 'buttons']:
                    if key in element_counts and element_counts[key] > report["components"][key]:
                        report["components"][key] = element_counts[key]
                        # If there's a significant difference, mark as dynamic
                        if element_counts[key] > report["components"][key] * 1.5:
                            report["is_dynamic"] = True
                
                # Enhanced Analysis: Interactive Elements
                report["interactive_elements"] = await detect_interactive_elements(page)
                
                # Enhanced Analysis: Visual Structure Analysis
                report["visual_structure"] = await capture_visual_analysis(page)
                
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
                
                # Scroll to bottom to trigger more content
                await page.evaluate("""() => {
                    window.scrollTo(0, document.body.scrollHeight);
                }""")
                
                await page.wait_for_timeout(2000)
                
                # Try clicking on "load more" buttons or pagination
                if report["interactive_elements"]:
                    for element in report["interactive_elements"][:2]:  # Limit to first 2
                        if "pagination" in element["type"] or "load" in element["text"].lower() or "more" in element["text"].lower():
                            try:
                                await page.click(element["selector"])
                                await page.wait_for_timeout(2000)  # Wait for content to load
                                app_logger.info(f"Clicked on interactive element: {element['text']}")
                            except Exception as e:
                                app_logger.warning(f"Failed to click interactive element: {e}")
                
                report["apis"] = api_requests[:10]  # Limit to 10 to avoid too much output
                
                await browser.close()
                
                app_logger.info(f"Dynamic analysis completed for: {url}")
                
        except Exception as e:
            error_details = traceback.format_exc()
            app_logger.error(f"Dynamic analysis error: {str(e)}\n{error_details}")
            report["warnings"].append(f"Dynamic analysis error: {str(e)}")
            
            # Fall back to static analysis determination of dynamic content
            report["is_dynamic"] = likely_dynamic
    else:
        app_logger.warning("Playwright not installed. Using static analysis to infer dynamic content.")
        report["warnings"].append("Dynamic analysis skipped: Playwright not installed")
        
        # Make a best guess about dynamic content based on static analysis
        report["is_dynamic"] = likely_dynamic
    
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
    
    # Enhanced recommendations
    if report["interactive_elements"]:
        interactive_types = set(element["type"] for element in report["interactive_elements"])
        if "pagination" in interactive_types:
            recs.append("Interactive pagination detected: Implement pagination navigation")
        if "button" in interactive_types and any("load" in el["text"].lower() or "more" in el["text"].lower() for el in report["interactive_elements"]):
            recs.append("Load more functionality detected: Implement click interaction to get all content")
    
    if report["content_patterns"]:
        pattern_with_highest_count = max(report["content_patterns"], key=lambda x: x["count"])
        recs.append(f"Repeating content pattern detected: Use selector '{pattern_with_highest_count['child_selector']}' for content items")
    
    if report["extraction_test_results"]:
        best_extraction = max(report["extraction_test_results"], key=lambda x: x["count"])
        recs.append(f"Extraction pattern '{best_extraction['pattern']}' found {best_extraction['count']} items with selector '{best_extraction['selector']}'")
    
    # Selector recommendations (original implementation)
    if report["selector_suggestions"]:
        for suggestion in report["selector_suggestions"]:
            recs.append(f"Selector suggestion: {suggestion}")
    
    report["recommendations"] = recs
    
    # Convert defaultdict to regular dict for JSON serialization
    report["content_hierarchy"] = dict(report["content_hierarchy"])
    
    execution_time = time.time() - start_time
    app_logger.info(f"Enhanced website analysis completed in {execution_time:.2f} seconds")
    
    # Cache the analysis results
    cache_analysis(url, report)
    
    # Convert the dictionary to a WebsiteAnalysis object
    return WebsiteAnalysis(**report)

# Update the original analyze_website function to use the enhanced version
async def analyze_website(url: str) -> 'WebsiteAnalysis':
    """
    Comprehensive webpage analyzer with parsing intelligence features.
    Now uses the enhanced analysis implementation with caching.
    
    Args:
        url: The URL of the website to analyze
        
    Returns:
        A WebsiteAnalysis object containing the analysis results
    """
    # Import cache service here to avoid circular imports
    from ..services.cache_service import get_cached_analysis, cache_analysis
    
    # Check if we have this URL cached
    cached_analysis = get_cached_analysis(url)
    if cached_analysis:
        # Convert dict to WebsiteAnalysis object
        from ..models.schemas import WebsiteAnalysis
        return WebsiteAnalysis(**cached_analysis)
    
    # No cache hit, perform full analysis
    analysis = await enhanced_website_analysis(url)
    
    # Cache the results for future use
    # Convert pydantic model to dict for caching if needed
    if hasattr(analysis, 'dict'):
        cache_analysis(url, analysis.dict())
    else:
        cache_analysis(url, dict(analysis))
    
    return analysis

# Synchronous wrapper for the async function (unchanged)
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
