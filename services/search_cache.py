"""
In-memory cache for search results with TTL support.
Reduces redundant API calls and improves response times.
"""
import hashlib
import time
import threading
import logging
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SearchCache:
    """Thread-safe in-memory cache for search results with TTL support."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize the search cache.
        
        Args:
            ttl_seconds: Time to live in seconds (default: 1 hour)
            max_size: Maximum number of entries to store (default: 1000)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self.lock = threading.RLock()
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.expirations = 0
        
        logger.info(f"SearchCache initialized with TTL={ttl_seconds}s, max_size={max_size}")
    
    def _generate_key(self, vertical: str, location: str, lat: float, lon: float, 
                     zoom: int, page: int) -> str:
        """
        Generate a deterministic cache key from search parameters.
        
        Args:
            vertical: Business type being searched
            location: Location name
            lat: Latitude
            lon: Longitude
            zoom: Map zoom level
            page: Result page number
            
        Returns:
            MD5 hash of the parameters
        """
        # Include coordinates for precision (rounded to 4 decimal places)
        key_str = f"{vertical}:{location}:{lat:.4f}:{lon:.4f}:{zoom}:{page}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, vertical: str, location: str, lat: float, lon: float, 
            zoom: int, page: int) -> Optional[Any]:
        """
        Retrieve a value from the cache if it exists and hasn't expired.
        
        Returns:
            Cached data if available and valid, None otherwise
        """
        key = self._generate_key(vertical, location, lat, lon, zoom, page)
        
        with self.lock:
            cache_entry = self.cache.get(key)
            
            if cache_entry is None:
                self.misses += 1
                logger.debug(f"Cache miss for key: {key[:8]}...")
                return None
            
            timestamp, data = cache_entry
            current_time = time.time()
            
            # Check if entry has expired
            if current_time - timestamp > self.ttl_seconds:
                # Remove expired entry
                del self.cache[key]
                self.expirations += 1
                self.misses += 1
                logger.debug(f"Cache entry expired for key: {key[:8]}...")
                return None
            
            # Move to end to maintain LRU order
            self.cache.move_to_end(key)
            self.hits += 1
            
            age_seconds = int(current_time - timestamp)
            logger.debug(f"Cache hit for key: {key[:8]}... (age: {age_seconds}s)")
            return data
    
    def set(self, vertical: str, location: str, lat: float, lon: float, 
            zoom: int, page: int, data: Any) -> None:
        """
        Store a value in the cache with current timestamp.
        
        Args:
            vertical: Business type being searched
            location: Location name
            lat: Latitude
            lon: Longitude
            zoom: Map zoom level
            page: Result page number
            data: Data to cache
        """
        key = self._generate_key(vertical, location, lat, lon, zoom, page)
        
        with self.lock:
            # Remove oldest entry if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Evicted oldest cache entry: {oldest_key[:8]}...")
            
            # Store with current timestamp
            self.cache[key] = (time.time(), data)
            logger.debug(f"Cached data for key: {key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "expirations": self.expirations,
                "hit_rate": round(hit_rate, 2),
                "ttl_seconds": self.ttl_seconds
            }
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        removed = 0
        
        with self.lock:
            # Create list of expired keys to avoid modifying dict during iteration
            expired_keys = [
                key for key, (timestamp, _) in self.cache.items()
                if current_time - timestamp > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self.cache[key]
                removed += 1
            
            if removed > 0:
                logger.info(f"Cleaned up {removed} expired cache entries")
        
        return removed