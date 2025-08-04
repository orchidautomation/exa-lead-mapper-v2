import requests
import logging
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod
import re
from urllib.parse import quote

from services.database import DatabaseManager

logger = logging.getLogger(__name__)

class GeocodingService(ABC):
    """Abstract base class for geocoding services."""
    
    @abstractmethod
    def geocode(self, location: str) -> Optional[Tuple[float, float]]:
        """Convert location to coordinates. Returns (latitude, longitude) or None."""
        pass
    
    @abstractmethod
    def batch_geocode(self, locations: List[str]) -> List[Optional[Tuple[float, float]]]:
        """Geocode multiple locations. Returns list of (lat, lon) tuples or None."""
        pass

class OpenMeteoGeocoding(GeocodingService):
    """Open-Meteo geocoding service (no API key required)."""
    
    BASE_URL = "https://geocoding-api.open-meteo.com/v1/search"
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def geocode(self, location: str) -> Optional[Tuple[float, float]]:
        """Geocode a single location using Open-Meteo."""
        try:
            # Check cache first
            cached = self.db.get_cached_location(location)
            if cached:
                logger.info(f"Using cached geocode for {location}")
                return cached
            
            # Parse location (city, state OR zip)
            location_parts = self._parse_location(location)
            if not location_parts:
                return None
            
            # Make API request
            params = {
                'name': location_parts['query'],
                'count': 1,
                'language': 'en',
                'format': 'json'
            }
            
            if location_parts.get('country'):
                params['country'] = location_parts['country']
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                result = data['results'][0]
                lat, lon = result['latitude'], result['longitude']
                
                # Cache the result
                self.db.cache_location(location, lat, lon, 'open-meteo')
                logger.info(f"Geocoded {location} to ({lat}, {lon})")
                
                return (lat, lon)
            
            logger.warning(f"No results found for location: {location}")
            return None
            
        except Exception as e:
            logger.error(f"Error geocoding {location}: {str(e)}")
            return None
    
    def batch_geocode(self, locations: List[str]) -> List[Optional[Tuple[float, float]]]:
        """Geocode multiple locations."""
        results = []
        for location in locations:
            results.append(self.geocode(location))
        return results
    
    def _parse_location(self, location: str) -> Optional[Dict]:
        """Parse location string into components."""
        # Remove extra whitespace
        location = location.strip()
        
        # Check if it's a US ZIP code
        zip_match = re.match(r'^(\d{5})(?:-\d{4})?$', location)
        if zip_match:
            return {
                'query': location,
                'country': 'US'
            }
        
        # Parse city, state format
        parts = [p.strip() for p in location.split(',')]
        
        if len(parts) == 2:
            city, state = parts
            # Check if state is a US state code
            if re.match(r'^[A-Z]{2}$', state.upper()):
                return {
                    'query': city,
                    'country': 'US'
                }
            else:
                return {
                    'query': f"{city}, {state}"
                }
        
        # Default: use the whole location as query
        return {'query': location}

class NominatimGeocoding(GeocodingService):
    """OpenStreetMap Nominatim geocoding service."""
    
    BASE_URL = "https://nominatim.openstreetmap.org/search"
    
    def __init__(self, db_manager: DatabaseManager, user_agent: str = "Mapper/1.0"):
        self.db = db_manager
        self.user_agent = user_agent
    
    def geocode(self, location: str) -> Optional[Tuple[float, float]]:
        """Geocode using Nominatim (respects 1 req/sec limit)."""
        try:
            # Check cache first
            cached = self.db.get_cached_location(location)
            if cached:
                logger.info(f"Using cached geocode for {location}")
                return cached
            
            headers = {'User-Agent': self.user_agent}
            params = {
                'q': location,
                'format': 'json',
                'limit': 1
            }
            
            # Add country bias for US locations
            if self._is_us_location(location):
                params['countrycodes'] = 'us'
            
            response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                
                # Cache the result
                self.db.cache_location(location, lat, lon, 'nominatim')
                logger.info(f"Geocoded {location} to ({lat}, {lon})")
                
                return (lat, lon)
            
            logger.warning(f"No results found for location: {location}")
            return None
            
        except Exception as e:
            logger.error(f"Error geocoding {location}: {str(e)}")
            return None
    
    def batch_geocode(self, locations: List[str]) -> List[Optional[Tuple[float, float]]]:
        """Geocode multiple locations with rate limiting."""
        import time
        results = []
        for i, location in enumerate(locations):
            if i > 0:
                time.sleep(1.1)  # Respect rate limit
            results.append(self.geocode(location))
        return results
    
    def _is_us_location(self, location: str) -> bool:
        """Check if location appears to be in the US."""
        # Check for US state codes
        if re.search(r'\b[A-Z]{2}\b', location.upper()):
            return True
        # Check for US ZIP codes
        if re.search(r'\b\d{5}(?:-\d{4})?\b', location):
            return True
        return False

class MapQuestGeocoding(GeocodingService):
    """MapQuest Open geocoding service."""
    
    BASE_URL = "https://www.mapquestapi.com/geocoding/v1"
    
    def __init__(self, db_manager: DatabaseManager, api_key: str):
        self.db = db_manager
        self.api_key = api_key
    
    def geocode(self, location: str) -> Optional[Tuple[float, float]]:
        """Geocode using MapQuest Open."""
        try:
            # Check cache first
            cached = self.db.get_cached_location(location)
            if cached:
                logger.info(f"Using cached geocode for {location}")
                return cached
            
            url = f"{self.BASE_URL}/address"
            params = {
                'key': self.api_key,
                'location': location,
                'maxResults': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('results') and data['results'][0].get('locations'):
                location_data = data['results'][0]['locations'][0]
                lat = location_data['latLng']['lat']
                lon = location_data['latLng']['lng']
                
                # Cache the result
                self.db.cache_location(location, lat, lon, 'mapquest')
                logger.info(f"Geocoded {location} to ({lat}, {lon})")
                
                return (lat, lon)
            
            logger.warning(f"No results found for location: {location}")
            return None
            
        except Exception as e:
            logger.error(f"Error geocoding {location}: {str(e)}")
            return None
    
    def batch_geocode(self, locations: List[str]) -> List[Optional[Tuple[float, float]]]:
        """Batch geocode using MapQuest."""
        try:
            url = f"{self.BASE_URL}/batch"
            data = {
                'key': self.api_key,
                'locations': locations
            }
            
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            
            results = []
            data = response.json()
            
            for i, result in enumerate(data.get('results', [])):
                if result.get('locations') and len(result['locations']) > 0:
                    location_data = result['locations'][0]
                    lat = location_data['latLng']['lat']
                    lon = location_data['latLng']['lng']
                    
                    # Cache each result
                    self.db.cache_location(locations[i], lat, lon, 'mapquest')
                    results.append((lat, lon))
                else:
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch geocoding: {str(e)}")
            # Fall back to individual geocoding
            return [self.geocode(loc) for loc in locations]

class GeocodingFactory:
    """Factory for creating geocoding service instances."""
    
    @staticmethod
    def create(service_name: str, db_manager: DatabaseManager, **kwargs) -> GeocodingService:
        """Create a geocoding service instance."""
        service_name = service_name.lower()
        
        if service_name == 'open-meteo':
            return OpenMeteoGeocoding(db_manager)
        elif service_name == 'nominatim':
            user_agent = kwargs.get('user_agent', 'Mapper/1.0')
            return NominatimGeocoding(db_manager, user_agent)
        elif service_name == 'mapquest':
            api_key = kwargs.get('api_key')
            if not api_key:
                raise ValueError("MapQuest requires an API key")
            return MapQuestGeocoding(db_manager, api_key)
        else:
            raise ValueError(f"Unknown geocoding service: {service_name}")