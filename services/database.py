"""
Simplified database manager for Business Mapper API v2.0
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import secrets
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations."""
    
    def __init__(self, db_path: str = "./data/mapper.db"):
        self.db_path = db_path
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database with required tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # API Keys table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    last_used_at TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            ''')
            
            # Search history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    api_key_id INTEGER,
                    search_params TEXT NOT NULL,
                    results_count INTEGER DEFAULT 0,
                    unique_results_count INTEGER DEFAULT 0,
                    credits_used INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
                )
            ''')
            
            # Search cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_cache (
                    cache_key TEXT PRIMARY KEY,
                    search_params TEXT NOT NULL,
                    results TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                )
            ''')
            
            # Location cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS location_cache (
                    location TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    formatted_address TEXT,
                    location_type TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Term performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS term_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    location_type TEXT,
                    search_count INTEGER DEFAULT 0,
                    total_results INTEGER DEFAULT 0,
                    unique_results INTEGER DEFAULT 0,
                    avg_contribution_rate REAL DEFAULT 0,
                    avg_cost_efficiency REAL DEFAULT 0,
                    avg_discovery_value REAL DEFAULT 0,
                    total_credits_used INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(term, location_type)
                )
            ''')
            
            logger.info("Database initialized successfully")
    
    # API Key Management
    def create_api_key(self, description: Optional[str] = None, 
                      expires_in_days: Optional[int] = None) -> Dict:
        """Create a new API key."""
        # Generate secure key
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:8]
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO api_keys (key_hash, key_prefix, description, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (key_hash, key_prefix, description, expires_at))
            
            return {
                "api_key": raw_key,
                "created_at": datetime.now(),
                "description": description,
                "expires_at": expires_at
            }
    
    def validate_api_key(self, api_key: str) -> Optional[int]:
        """Validate an API key and return its ID if valid."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, expires_at FROM api_keys 
                WHERE key_hash = ? AND is_active = 1
            ''', (key_hash,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Check expiration
            if row['expires_at']:
                expires_at = datetime.fromisoformat(row['expires_at'])
                if expires_at < datetime.now():
                    return None
            
            # Update usage
            cursor.execute('''
                UPDATE api_keys 
                SET last_used_at = CURRENT_TIMESTAMP, 
                    usage_count = usage_count + 1
                WHERE id = ?
            ''', (row['id'],))
            
            return row['id']
    
    # Search History
    def log_search(self, session_id: str, api_key_id: Optional[int], 
                   search_params: Dict, results_count: int, 
                   unique_results_count: int, credits_used: int, cost: float):
        """Log a search to history."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO search_history 
                (session_id, api_key_id, search_params, results_count, 
                 unique_results_count, credits_used, cost)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, api_key_id, json.dumps(search_params), 
                  results_count, unique_results_count, credits_used, cost))
    
    # Cache Management
    def get_search_cache(self, cache_key: str) -> Optional[Dict]:
        """Get cached search results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT results, metadata FROM search_cache
                WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
            ''', (cache_key,))
            
            row = cursor.fetchone()
            if row:
                # Update hit count
                cursor.execute('''
                    UPDATE search_cache SET hit_count = hit_count + 1
                    WHERE cache_key = ?
                ''', (cache_key,))
                
                return {
                    "results": json.loads(row['results']),
                    "metadata": json.loads(row['metadata']) if row['metadata'] else {}
                }
            return None
    
    def set_search_cache(self, cache_key: str, results: Any, 
                        metadata: Optional[Dict] = None, ttl_hours: int = 24):
        """Set search cache."""
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO search_cache 
                (cache_key, search_params, results, metadata, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (cache_key, "", json.dumps(results), 
                  json.dumps(metadata) if metadata else None, expires_at))
    
    # Location Cache
    def get_cached_location(self, location: str) -> Optional[Tuple[float, float]]:
        """Get cached geocoding result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT latitude, longitude FROM location_cache
                WHERE location = ?
            ''', (location.lower(),))
            
            row = cursor.fetchone()
            if row:
                return (row['latitude'], row['longitude'])
            return None
    
    def cache_location(self, location: str, lat: float, lon: float, 
                      location_type: Optional[str] = None):
        """Cache geocoding result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO location_cache 
                (location, latitude, longitude, location_type)
                VALUES (?, ?, ?, ?)
            ''', (location.lower(), lat, lon, location_type))
    
    # Term Performance
    def update_term_performance(self, term: str, location_type: str, 
                               contribution_data: Dict):
        """Update term performance metrics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute('''
                SELECT search_count FROM term_performance
                WHERE term = ? AND location_type = ?
            ''', (term, location_type))
            
            row = cursor.fetchone()
            if row:
                # Update existing
                cursor.execute('''
                    UPDATE term_performance
                    SET search_count = search_count + 1,
                        total_results = total_results + ?,
                        unique_results = unique_results + ?,
                        avg_contribution_rate = 
                            (avg_contribution_rate * search_count + ?) / (search_count + 1),
                        avg_cost_efficiency = 
                            (avg_cost_efficiency * search_count + ?) / (search_count + 1),
                        avg_discovery_value = 
                            (avg_discovery_value * search_count + ?) / (search_count + 1),
                        total_credits_used = total_credits_used + ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE term = ? AND location_type = ?
                ''', (contribution_data['total_results'], 
                      contribution_data['unique_results'],
                      contribution_data['contribution_rate'],
                      contribution_data['cost_efficiency'],
                      contribution_data['discovery_value'],
                      contribution_data.get('credits_used', 3),
                      term, location_type))
            else:
                # Insert new
                cursor.execute('''
                    INSERT INTO term_performance
                    (term, location_type, search_count, total_results, unique_results,
                     avg_contribution_rate, avg_cost_efficiency, avg_discovery_value,
                     total_credits_used)
                    VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)
                ''', (term, location_type, 
                      contribution_data['total_results'],
                      contribution_data['unique_results'],
                      contribution_data['contribution_rate'],
                      contribution_data['cost_efficiency'],
                      contribution_data['discovery_value'],
                      contribution_data.get('credits_used', 3)))
    
    # Cleanup
    def cleanup_expired_cache(self):
        """Remove expired cache entries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM search_cache WHERE expires_at < CURRENT_TIMESTAMP')
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired cache entries")