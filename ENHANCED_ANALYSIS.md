# Enhanced Website Analysis Module

This update introduces several advanced improvements to the Auto Web Scraper's website analysis capabilities. These enhancements are designed to provide better accuracy, performance, and reliability when scraping websites.

## Key Improvements

### 1. Enhanced Content Analysis

- **Content Pattern Detection**: Identifies repeating patterns in website structure to better target content elements
- **DOM Visualization**: Creates a visual representation of the DOM hierarchy to understand page structure
- **Interactive Element Detection**: Identifies buttons, pagination controls, and other interactive elements
- **Selector Strategy Generation**: Creates multiple selector strategies with reliability scores

### 2. Improved Dynamic Content Handling

- **Better Dynamic Content Detection**: More sophisticated algorithms to determine if a site uses JavaScript
- **Visual Analysis**: Captures screenshots and analyzes visual structure (when Playwright is available)
- **Interactive Testing**: Tries clicking on load more buttons to test content loading behavior

### 3. Performance Optimizations

- **Caching System**: Caches analysis results to avoid redundant processing of frequently scraped sites
- **Content-Specific Code Templates**: Generates specialized code templates based on site type
- **Smart Selector Prioritization**: Prioritizes more reliable selectors to reduce failures

### 4. Error Handling Improvements

- **Extraction Testing**: Tests common patterns to identify what works for specific sites
- **Detailed Error Analysis**: Provides more specific guidance when scraping fails
- **Smart Retries**: More intelligent retry strategies for failed scraping attempts

## How to Use

The enhanced analysis is now integrated directly into the existing workflow. No changes to your usage are required - the system will automatically take advantage of these improvements.

### For Developers

If you want to utilize specific features for your own development:

```python
# Import the enhanced analysis functions
from app.services.enhanced_llm import create_enhanced_scraping_prompt
from app.services.website_analyzer import analyze_website

# Analyze a website
analysis = await analyze_website("https://example.com")

# Access enhanced data
content_patterns = analysis.content_patterns
interactive_elements = analysis.interactive_elements
selector_strategies = analysis.selector_strategies
```

## Cache Management

The new caching system stores analysis results to improve performance. Cache entries expire after 24 hours by default.

To manage the cache:

```python
from app.services.cache_service import clear_cache, get_cache_stats

# Clear entire cache
clear_cache()

# Clear cache for specific URL
clear_cache("https://example.com")

# Get cache statistics
stats = get_cache_stats()
print(f"Cache entries: {stats['total_entries']}")
print(f"Total size: {stats['total_size_bytes'] / 1024:.2f} KB")
```

## Requirements

These enhancements work best with the following dependencies:

- `playwright`: For dynamic content analysis and interactive testing
- `langdetect`: For content language detection

Install with:
```bash
pip install playwright langdetect
python -m playwright install chromium
```