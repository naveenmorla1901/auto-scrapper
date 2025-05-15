"""
Implements a simple caching mechanism for website analysis results
to improve performance for repeated scraping of the same sites.
"""
import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pickle
from ..utils.logger import app_logger

# Create cache directory if it doesn't exist
CACHE_DIR = Path("cache/website_analysis")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache expiration time (24 hours by default)
CACHE_EXPIRATION = 86400  # seconds

def get_cache_key(url: str) -> str:
    """
    Generate a unique cache key for a URL
    
    Args:
        url: The website URL
        
    Returns:
        A hash-based cache key
    """
    return hashlib.md5(url.encode()).hexdigest()

def get_cached_analysis(url: str) -> Optional[Dict[str, Any]]:
    """
    Get cached website analysis if available
    
    Args:
        url: URL of the website
        
    Returns:
        Cached analysis or None
    """
    cache_key = get_cache_key(url)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                
            # Check if cache is fresh
            cache_time = cached_data.get("cache_time", 0)
            if time.time() - cache_time < CACHE_EXPIRATION:
                app_logger.info(f"Using cached analysis for: {url}")
                return cached_data.get("analysis")
            else:
                app_logger.info(f"Cache expired for: {url}")
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
    cache_key = get_cache_key(url)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    
    try:
        with open(cache_file, "wb") as f:
            pickle.dump({
                "analysis": analysis,
                "cache_time": time.time(),
                "url": url
            }, f)
        app_logger.info(f"Cached analysis for: {url}")
    except Exception as e:
        app_logger.warning(f"Failed to cache analysis: {e}")

def clear_cache(url: Optional[str] = None) -> None:
    """
    Clear the analysis cache
    
    Args:
        url: Optional URL to clear specific cache entry.
             If None, clears all cache entries.
    """
    if url:
        # Clear specific cache entry
        cache_key = get_cache_key(url)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                os.remove(cache_file)
                app_logger.info(f"Cleared cache for: {url}")
            except Exception as e:
                app_logger.warning(f"Failed to clear cache for {url}: {e}")
    else:
        # Clear all cache entries
        try:
            cache_files = list(CACHE_DIR.glob("*.pkl"))
            for cache_file in cache_files:
                os.remove(cache_file)
            app_logger.info(f"Cleared {len(cache_files)} cache entries")
        except Exception as e:
            app_logger.warning(f"Failed to clear cache: {e}")

def remove_expired_cache() -> int:
    """
    Remove expired cache entries
    
    Returns:
        Number of removed entries
    """
    removed = 0
    try:
        for cache_file in CACHE_DIR.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                
                cache_time = cached_data.get("cache_time", 0)
                if time.time() - cache_time >= CACHE_EXPIRATION:
                    os.remove(cache_file)
                    removed += 1
            except Exception:
                # If we can't read the file, remove it
                os.remove(cache_file)
                removed += 1
        
        if removed > 0:
            app_logger.info(f"Removed {removed} expired cache entries")
        
        return removed
    except Exception as e:
        app_logger.warning(f"Failed to clean cache: {e}")
        return 0

def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics
    
    Returns:
        Dictionary with cache statistics
    """
    stats = {
        "total_entries": 0,
        "expired_entries": 0,
        "total_size_bytes": 0,
        "average_age_hours": 0
    }
    
    try:
        cache_files = list(CACHE_DIR.glob("*.pkl"))
        stats["total_entries"] = len(cache_files)
        
        total_age = 0
        current_time = time.time()
        
        for cache_file in cache_files:
            stats["total_size_bytes"] += cache_file.stat().st_size
            
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                
                cache_time = cached_data.get("cache_time", 0)
                age = current_time - cache_time
                total_age += age
                
                if age >= CACHE_EXPIRATION:
                    stats["expired_entries"] += 1
            except Exception:
                # Count as expired if we can't read it
                stats["expired_entries"] += 1
        
        if stats["total_entries"] > 0:
            stats["average_age_hours"] = (total_age / stats["total_entries"]) / 3600
        
        return stats
    except Exception as e:
        app_logger.warning(f"Failed to get cache stats: {e}")
        return stats