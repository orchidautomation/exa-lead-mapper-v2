import sqlite3
import logging
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationCache:
    """
    SQLite-based caching system for AI validation results.
    
    Stores validation results with TTL to minimize expensive AI API calls.
    Designed for high hit rates after an initial learning period.
    """
    
    def __init__(self, db_path: str = "validation_cache.db", ttl_days: int = 30):
        """
        Initialize validation cache.
        
        Args:
            db_path: Path to SQLite database file
            ttl_days: Time-to-live for cached results in days
        """
        self.db_path = db_path
        self.ttl_days = ttl_days
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Stats tracking
        self._hit_count = 0
        self._miss_count = 0
        self._session_start = time.time()
        
        logger.info(f"Validation cache initialized: {db_path} (TTL: {ttl_days} days)")
    
    def _init_database(self):
        """Initialize the cache database table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS validation_cache (
                        cache_key TEXT PRIMARY KEY,
                        is_match BOOLEAN NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        last_accessed REAL
                    )
                """)
                
                # Create index for efficient expiration cleanup
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at 
                    ON validation_cache(expires_at)
                """)
                
                # Create index for analytics
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at 
                    ON validation_cache(created_at)
                """)
                
                # City classification cache table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS city_classifications (
                        cache_key TEXT PRIMARY KEY,
                        city_name TEXT NOT NULL,
                        density_category TEXT NOT NULL,
                        confidence REAL,
                        reasoning TEXT,
                        population_estimate INTEGER,
                        country TEXT,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        last_accessed REAL
                    )
                """)
                
                # Create index for efficient expiration cleanup
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_city_expires_at 
                    ON city_classifications(expires_at)
                """)
                
                # Create index for city name lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_city_name 
                    ON city_classifications(city_name)
                """)
                
                conn.commit()
                logger.debug("Validation cache database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize validation cache database: {str(e)}")
            raise
    
    def get_validation(self, cache_key: str) -> Optional[bool]:
        """
        Get cached validation result.
        
        Args:
            cache_key: The cache key for the validation
            
        Returns:
            Boolean validation result if found and not expired, None otherwise
        """
        try:
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT is_match, expires_at 
                    FROM validation_cache 
                    WHERE cache_key = ? AND expires_at > ?
                """, (cache_key, current_time))
                
                result = cursor.fetchone()
                
                if result:
                    is_match, expires_at = result
                    
                    # Update hit count and last accessed
                    conn.execute("""
                        UPDATE validation_cache 
                        SET hit_count = hit_count + 1, last_accessed = ?
                        WHERE cache_key = ?
                    """, (current_time, cache_key))
                    
                    conn.commit()
                    
                    self._hit_count += 1
                    logger.debug(f"Cache HIT: {cache_key[:8]}... -> {bool(is_match)}")
                    return bool(is_match)
                else:
                    self._miss_count += 1
                    logger.debug(f"Cache MISS: {cache_key[:8]}...")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting validation from cache: {str(e)}")
            self._miss_count += 1
            return None
    
    def set_validation(self, cache_key: str, is_match: bool) -> bool:
        """
        Store validation result in cache.
        
        Args:
            cache_key: The cache key for the validation
            is_match: The validation result to cache
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            current_time = time.time()
            expires_at = current_time + self.ttl_seconds
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO validation_cache 
                    (cache_key, is_match, created_at, expires_at, hit_count, last_accessed)
                    VALUES (?, ?, ?, ?, 0, ?)
                """, (cache_key, is_match, current_time, expires_at, current_time))
                
                conn.commit()
                
                logger.debug(f"Cache SET: {cache_key[:8]}... -> {is_match}")
                return True
                
        except Exception as e:
            logger.error(f"Error setting validation in cache: {str(e)}")
            return False
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        try:
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM validation_cache 
                    WHERE expires_at <= ?
                """, (current_time,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired cache entries")
                
                # Also clean expired city classifications
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM city_classifications WHERE expires_at <= ?",
                        (current_time,)
                    )
                    city_deleted = cursor.rowcount
                    conn.commit()
                    
                    if city_deleted > 0:
                        logger.info(f"Cleaned up {city_deleted} expired city classification entries")
                
                return deleted_count + city_deleted
                
        except Exception as e:
            logger.error(f"Error cleaning up expired cache entries: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            current_time = time.time()
            session_duration = current_time - self._session_start
            
            with sqlite3.connect(self.db_path) as conn:
                # Total entries
                cursor = conn.execute("SELECT COUNT(*) FROM validation_cache")
                total_entries = cursor.fetchone()[0]
                
                # Active (non-expired) entries
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM validation_cache 
                    WHERE expires_at > ?
                """, (current_time,))
                active_entries = cursor.fetchone()[0]
                
                # Expired entries
                expired_entries = total_entries - active_entries
                
                # Hit rate calculation
                total_requests = self._hit_count + self._miss_count
                hit_rate = (self._hit_count / total_requests * 100) if total_requests > 0 else 0
                
                # Most popular entries
                cursor = conn.execute("""
                    SELECT cache_key, hit_count 
                    FROM validation_cache 
                    WHERE expires_at > ?
                    ORDER BY hit_count DESC 
                    LIMIT 5
                """, (current_time,))
                popular_entries = cursor.fetchall()
                
                # Recent activity
                recent_cutoff = current_time - (24 * 60 * 60)  # Last 24 hours
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM validation_cache 
                    WHERE created_at > ?
                """, (recent_cutoff,))
                recent_entries = cursor.fetchone()[0]
                
                return {
                    'total_entries': total_entries,
                    'active_entries': active_entries,
                    'expired_entries': expired_entries,
                    'recent_entries_24h': recent_entries,
                    'hit_count': self._hit_count,
                    'miss_count': self._miss_count,
                    'hit_rate_percent': round(hit_rate, 2),
                    'session_duration_minutes': round(session_duration / 60, 2),
                    'popular_entries': [
                        {'key_prefix': key[:12], 'hits': hits} 
                        for key, hits in popular_entries
                    ],
                    'ttl_days': self.ttl_days,
                    'db_path': self.db_path
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {
                'error': str(e),
                'hit_count': self._hit_count,
                'miss_count': self._miss_count
            }
    
    def clear_cache(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM validation_cache")
                conn.commit()
                
                logger.info("Validation cache cleared")
                
                # Reset stats
                self._hit_count = 0
                self._miss_count = 0
                self._session_start = time.time()
                
                return True
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def optimize_database(self) -> bool:
        """
        Optimize the database by running VACUUM and ANALYZE.
        
        Returns:
            True if optimized successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clean up expired entries first
                self.cleanup_expired()
                
                # Optimize database
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                
                logger.info("Validation cache database optimized")
                return True
                
        except Exception as e:
            logger.error(f"Error optimizing cache database: {str(e)}")
            return False
    
    def export_popular_patterns(self, limit: int = 100) -> Dict:
        """
        Export most popular validation patterns for analysis.
        
        Args:
            limit: Number of top patterns to export
            
        Returns:
            Dictionary with popular validation patterns
        """
        try:
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT cache_key, is_match, hit_count, created_at, last_accessed
                    FROM validation_cache 
                    WHERE expires_at > ? AND hit_count > 0
                    ORDER BY hit_count DESC 
                    LIMIT ?
                """, (current_time, limit))
                
                patterns = []
                for row in cursor.fetchall():
                    cache_key, is_match, hit_count, created_at, last_accessed = row
                    patterns.append({
                        'cache_key_prefix': cache_key[:16],
                        'is_match': bool(is_match),
                        'hit_count': hit_count,
                        'created_date': datetime.fromtimestamp(created_at).isoformat(),
                        'last_accessed_date': datetime.fromtimestamp(last_accessed).isoformat() if last_accessed else None
                    })
                
                return {
                    'patterns': patterns,
                    'exported_at': datetime.now().isoformat(),
                    'total_patterns': len(patterns)
                }
                
        except Exception as e:
            logger.error(f"Error exporting popular patterns: {str(e)}")
            return {'error': str(e), 'patterns': []}
    
    # City Classification Cache Methods
    def get_city_classification(self, cache_key: str):
        """
        Get cached city classification result.
        
        Args:
            cache_key: The cache key for the city classification
            
        Returns:
            Tuple of (density_category, confidence, reasoning, population_estimate, country) 
            if found and not expired, None otherwise
        """
        try:
            import time
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT density_category, confidence, reasoning, population_estimate, country
                    FROM city_classifications 
                    WHERE cache_key = ? AND expires_at > ?
                """, (cache_key, current_time))
                
                result = cursor.fetchone()
                
                if result:
                    # Update hit count and last accessed
                    conn.execute("""
                        UPDATE city_classifications 
                        SET hit_count = hit_count + 1, last_accessed = ?
                        WHERE cache_key = ?
                    """, (current_time, cache_key))
                    
                    conn.commit()
                    
                    logger.debug(f"City classification cache HIT: {cache_key[:8]}...")
                    return tuple(result)
                else:
                    logger.debug(f"City classification cache MISS: {cache_key[:8]}...")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting city classification from cache: {str(e)}")
            return None
    
    def set_city_classification(self, cache_key: str, city_name: str, density_category: str, 
                              confidence: float, reasoning: str, population_estimate=None,
                              country=None, ttl_days: int = 365) -> bool:
        """
        Store city classification result in cache.
        
        Args:
            cache_key: The cache key for the city classification
            city_name: Full city name
            density_category: The density classification (dense_urban, urban, suburban, rural)
            confidence: Confidence score (0.0 to 1.0)
            reasoning: AI reasoning or fallback explanation
            population_estimate: Optional population estimate
            country: Optional country information
            ttl_days: Time-to-live in days (default 1 year)
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            import time
            current_time = time.time()
            expires_at = current_time + (ttl_days * 24 * 60 * 60)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO city_classifications 
                    (cache_key, city_name, density_category, confidence, reasoning, 
                     population_estimate, country, created_at, expires_at, hit_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (cache_key, city_name, density_category, confidence, reasoning,
                      population_estimate, country, current_time, expires_at, current_time))
                
                conn.commit()
                
                logger.debug(f"City classification cache SET: {cache_key[:8]}... -> {density_category}")
                return True
                
        except Exception as e:
            logger.error(f"Error setting city classification in cache: {str(e)}")
            return False
    
    def get_city_cache_stats(self) -> Dict:
        """
        Get city classification cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            import time
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                # Total entries
                cursor = conn.execute("SELECT COUNT(*) FROM city_classifications")
                total_entries = cursor.fetchone()[0]
                
                # Active (non-expired) entries
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM city_classifications 
                    WHERE expires_at > ?
                """, (current_time,))
                active_entries = cursor.fetchone()[0]
                
                # Category distribution
                cursor = conn.execute("""
                    SELECT density_category, COUNT(*) 
                    FROM city_classifications 
                    WHERE expires_at > ?
                    GROUP BY density_category
                """, (current_time,))
                category_distribution = dict(cursor.fetchall())
                
                # Most popular cities
                cursor = conn.execute("""
                    SELECT city_name, hit_count 
                    FROM city_classifications 
                    WHERE expires_at > ?
                    ORDER BY hit_count DESC 
                    LIMIT 10
                """, (current_time,))
                popular_cities = cursor.fetchall()
                
                return {
                    'city_cache_total_entries': total_entries,
                    'city_cache_active_entries': active_entries,
                    'city_cache_category_distribution': category_distribution,
                    'city_cache_popular_cities': [
                        {'city': city, 'hits': hits} for city, hits in popular_cities
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting city cache stats: {str(e)}")
            return {
                'error': str(e),
                'city_cache_total_entries': 0,
                'city_cache_active_entries': 0
            }