import math
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LocationType(Enum):
    """Types of locations based on business density."""
    DENSE_URBAN = "dense_urban"      # High business density
    URBAN = "urban"                  # Medium-high business density  
    SUBURBAN = "suburban"            # Medium business density
    RURAL = "rural"                  # Low business density
    UNKNOWN = "unknown"              # Not yet determined

@dataclass
class LocationCluster:
    """Represents a cluster of nearby locations."""
    centroid_lat: float
    centroid_lon: float
    centroid_name: str
    locations: List[Tuple[str, float, float]]
    radius_miles: float
    recommended_zoom: int
    location_type: LocationType
    
    @property
    def location_count(self) -> int:
        return len(self.locations)

class GeographicOptimizer:
    """Handles geographic optimization for search efficiency."""
    
    # Distance thresholds in miles
    OVERLAP_THRESHOLDS = {
        LocationType.DENSE_URBAN: 8,    # Tight clustering in cities
        LocationType.URBAN: 12,         # Medium clustering 
        LocationType.SUBURBAN: 15,      # Wider clustering
        LocationType.RURAL: 20,         # Wide clustering
        LocationType.UNKNOWN: 15        # Default conservative
    }
    
    # Optimal zoom levels by location type
    ZOOM_LEVELS = {
        LocationType.DENSE_URBAN: 15,   # Focused search
        LocationType.URBAN: 14,         # Balanced
        LocationType.SUBURBAN: 13,      # Wider coverage
        LocationType.RURAL: 12,         # Wide coverage
        LocationType.UNKNOWN: 14        # Default balanced
    }
    
    # Location-aware pagination rules
    PAGINATION_RULES = {
        LocationType.DENSE_URBAN: {
            'min_pages': 2,              # Always search at least 2 pages
            'max_pages': 6,              # Cap at 6 pages max
            'stop_threshold': 5,         # Stop if <5 new unique results
            'duplicate_threshold': 0.40, # Stop if >40% duplicates
            'description': 'Dense urban areas have many businesses, pagination often valuable'
        },
        LocationType.URBAN: {
            'min_pages': 2,              # Always search at least 2 pages
            'max_pages': 4,              # Cap at 4 pages
            'stop_threshold': 6,         # Stop if <6 new unique results
            'duplicate_threshold': 0.45, # Stop if >45% duplicates
            'description': 'Urban areas have good business density, moderate pagination'
        },
        LocationType.SUBURBAN: {
            'min_pages': 1,              # May stop after 1 page
            'max_pages': 3,              # Cap at 3 pages
            'stop_threshold': 8,         # Stop if <8 new unique results
            'duplicate_threshold': 0.50, # Stop if >50% duplicates
            'description': 'Suburban areas have fewer businesses, limited pagination value'
        },
        LocationType.RURAL: {
            'min_pages': 1,              # May stop after 1 page
            'max_pages': 2,              # Cap at 2 pages
            'stop_threshold': 10,        # Stop if <10 new unique results
            'duplicate_threshold': 0.60, # Stop if >60% duplicates
            'description': 'Rural areas have sparse businesses, early stopping likely'
        },
        LocationType.UNKNOWN: {
            'min_pages': 1,              # Conservative approach
            'max_pages': 3,              # Conservative cap
            'stop_threshold': 8,         # Conservative threshold
            'duplicate_threshold': 0.45, # Conservative duplicate limit
            'description': 'Unknown location type, use conservative pagination'
        }
    }
    
    def __init__(self, db_manager=None):
        self.location_intelligence = {}  # Cache location characteristics
        
        # Initialize AI city classifier if database manager is provided
        self.ai_city_classifier = None
        if db_manager:
            try:
                from services.city_classifier import AICityClassifier
                self.ai_city_classifier = AICityClassifier(cache_manager=db_manager)
                if self.ai_city_classifier.enabled:
                    logger.info("🌍 AI city classification enabled in GeographicOptimizer")
                else:
                    logger.info("ℹ️  AI city classification available but not enabled (missing Groq API key)")
            except ImportError as e:
                logger.warning(f"AI city classification not available: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to initialize AI city classifier: {str(e)}")
                self.ai_city_classifier = None
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points in miles.
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            Distance in miles
        """
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in miles
        earth_radius_miles = 3959
        
        return earth_radius_miles * c
    
    def calculate_centroid(self, locations: List[Tuple[str, float, float]]) -> Tuple[float, float, str]:
        """
        Calculate the geographic centroid of a list of locations.
        
        Args:
            locations: List of (name, latitude, longitude) tuples
            
        Returns:
            Tuple of (centroid_lat, centroid_lon, combined_name)
        """
        if not locations:
            raise ValueError("Cannot calculate centroid of empty location list")
        
        if len(locations) == 1:
            return locations[0][1], locations[0][2], locations[0][0]
        
        # Calculate centroid
        total_lat = sum(loc[1] for loc in locations)
        total_lon = sum(loc[2] for loc in locations)
        
        centroid_lat = total_lat / len(locations)
        centroid_lon = total_lon / len(locations)
        
        # Create combined name
        location_names = [loc[0] for loc in locations]
        if len(location_names) <= 3:
            combined_name = " + ".join(location_names)
        else:
            combined_name = f"{location_names[0]} + {len(location_names)-1} others"
        
        return centroid_lat, centroid_lon, combined_name
    
    # Comprehensive location classification database
    LOCATION_CLASSIFICATIONS = {
        'dense_urban': [
            # Major US Cities
            'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
            'san antonio', 'san diego', 'dallas', 'san jose', 'austin', 'jacksonville',
            'fort worth', 'columbus', 'charlotte', 'san francisco', 'indianapolis',
            'seattle', 'denver', 'washington', 'boston', 'el paso', 'detroit', 'nashville',
            'portland', 'memphis', 'oklahoma city', 'las vegas', 'louisville', 'baltimore',
            'milwaukee', 'albuquerque', 'tucson', 'fresno', 'mesa', 'sacramento',
            'atlanta', 'kansas city', 'colorado springs', 'raleigh', 'omaha', 'miami',
            'long beach', 'virginia beach', 'oakland', 'minneapolis', 'tulsa', 'tampa',
            'arlington', 'new orleans', 'wichita', 'cleveland', 'bakersfield', 'honolulu'
        ],
        'urban': [
            # Mid-size cities
            'anaheim', 'santa ana', 'riverside', 'stockton', 'corpus christi', 'lexington',
            'anchorage', 'cincinnati', 'henderson', 'greensboro', 'plano', 'newark',
            'lincoln', 'buffalo', 'jersey city', 'chula vista', 'fort wayne', 'orlando',
            'st. petersburg', 'chandler', 'laredo', 'norfolk', 'durham', 'madison',
            'lubbock', 'irvine', 'winston-salem', 'glendale', 'garland', 'hialeah',
            'reno', 'chesapeake', 'gilbert', 'baton rouge', 'irving', 'scottsdale',
            'north las vegas', 'fremont', 'boise', 'richmond', 'san bernardino', 'birmingham'
        ],
        'suburban': [
            # Suburban cities and large towns
            'frisco', 'round rock', 'pearland', 'sugar land', 'allen', 'mckinney',
            'carrollton', 'richardson', 'lewisville', 'flower mound', 'cedar park',
            'georgetown', 'pflugerville', 'katy', 'missouri city', 'pasadena', 'mesquite',
            'denton', 'abilene', 'beaumont', 'tyler', 'waco', 'college station', 'killeen',
            'longview', 'bryan', 'pharr', 'carrollton', 'league city', 'baytown', 'conroe'
        ]
    }
    
    # Comprehensive ZIP code classifications
    ZIP_CLASSIFICATIONS = {
        # New York Metro
        **{zip_code: 'dense_urban' for zip_code in range(10001, 10299)},  # Manhattan
        **{zip_code: 'dense_urban' for zip_code in range(11201, 11256)},  # Brooklyn
        **{zip_code: 'urban' for zip_code in range(11301, 11697)},        # Queens
        **{zip_code: 'suburban' for zip_code in range(10701, 10710)},     # Westchester
        
        # Los Angeles Metro  
        **{zip_code: 'dense_urban' for zip_code in range(90001, 90089)},  # LA City
        **{zip_code: 'dense_urban' for zip_code in range(90210, 90213)},  # Beverly Hills
        **{zip_code: 'urban' for zip_code in range(90201, 90210)},        # LA County
        **{zip_code: 'suburban' for zip_code in range(91001, 91199)},     # Pasadena area
        
        # Chicago Metro
        **{zip_code: 'dense_urban' for zip_code in range(60601, 60661)},  # Downtown Chicago
        **{zip_code: 'urban' for zip_code in range(60007, 60199)},        # Chicago suburbs
        **{zip_code: 'suburban' for zip_code in range(60401, 60499)},     # Outer suburbs
        
        # Houston Metro
        **{zip_code: 'dense_urban' for zip_code in range(77001, 77099)},  # Houston city
        **{zip_code: 'urban' for zip_code in range(77301, 77399)},        # Houston metro
        **{zip_code: 'suburban' for zip_code in range(77401, 77499)},     # Sugar Land area
        
        # Austin Metro
        **{zip_code: 'dense_urban' for zip_code in range(78701, 78759)},  # Austin city
        **{zip_code: 'urban' for zip_code in range(78613, 78653)},        # Austin metro
        **{zip_code: 'suburban' for zip_code in range(78660, 78680)},     # Round Rock area
        
        # Dallas Metro
        **{zip_code: 'dense_urban' for zip_code in range(75201, 75299)},  # Dallas city
        **{zip_code: 'urban' for zip_code in range(75001, 75099)},        # Dallas metro
        **{zip_code: 'suburban' for zip_code in range(75101, 75199)},     # Plano/Frisco
        
        # San Francisco Bay Area
        **{zip_code: 'dense_urban' for zip_code in range(94102, 94134)},  # SF city
        **{zip_code: 'urban' for zip_code in range(94301, 94309)},        # Palo Alto
        **{zip_code: 'suburban' for zip_code in range(95014, 95070)},     # South Bay suburbs
        
        # Miami Metro
        **{zip_code: 'dense_urban' for zip_code in range(33101, 33199)},  # Miami-Dade
        **{zip_code: 'urban' for zip_code in range(33301, 33399)},        # Broward
        **{zip_code: 'suburban' for zip_code in range(33401, 33499)},     # Palm Beach
        
        # Seattle Metro
        **{zip_code: 'dense_urban' for zip_code in range(98101, 98199)},  # Seattle city
        **{zip_code: 'urban' for zip_code in range(98001, 98099)},        # King County
        **{zip_code: 'suburban' for zip_code in range(98201, 98299)},     # Snohomish County
        
        # Boston Metro
        **{zip_code: 'dense_urban' for zip_code in range(2101, 2137)},    # Boston city
        **{zip_code: 'urban' for zip_code in range(2138, 2199)},          # Boston metro
        **{zip_code: 'suburban' for zip_code in range(1701, 1899)},       # Outer suburbs
        
        # Phoenix Metro
        **{zip_code: 'dense_urban' for zip_code in range(85001, 85099)},  # Phoenix city
        **{zip_code: 'urban' for zip_code in range(85201, 85299)},        # Mesa/Tempe
        **{zip_code: 'suburban' for zip_code in range(85301, 85399)},     # Scottsdale area
    }
    
    def classify_location_type(self, location_name: str, historical_data: Optional[Dict] = None) -> LocationType:
        """
        Classify location type based on comprehensive database and historical data.
        
        Args:
            location_name: Name of the location
            historical_data: Previous search results for this location
            
        Returns:
            LocationType enum value
        """
        # Check cache first
        if location_name in self.location_intelligence:
            return self.location_intelligence[location_name]['type']
        
        # Use historical data if available (most accurate)
        if historical_data:
            avg_results_per_page = historical_data.get('avg_results_per_page', 0)
            if avg_results_per_page >= 19:
                return LocationType.DENSE_URBAN
            elif avg_results_per_page >= 15:
                return LocationType.URBAN
            elif avg_results_per_page >= 10:
                return LocationType.SUBURBAN
            elif avg_results_per_page > 0:
                return LocationType.RURAL
        
        # ZIP code classification (most precise for ZIP inputs)
        if location_name.isdigit() and len(location_name) == 5:
            zip_code = int(location_name)
            zip_type = self.ZIP_CLASSIFICATIONS.get(zip_code)
            if zip_type == 'dense_urban':
                return LocationType.DENSE_URBAN
            elif zip_type == 'urban':
                return LocationType.URBAN
            elif zip_type == 'suburban':
                return LocationType.SUBURBAN
            # If ZIP not in database, fall through to city name classification
        
        # City name classification
        location_lower = location_name.lower()
        
        # Remove common suffixes for better matching
        location_clean = location_lower.replace(', tx', '').replace(', ca', '').replace(', ny', '')
        location_clean = location_clean.replace(' city', '').strip()
        
        # Check dense urban cities
        for city in self.LOCATION_CLASSIFICATIONS['dense_urban']:
            if city in location_clean or location_clean in city:
                return LocationType.DENSE_URBAN
        
        # Check urban cities
        for city in self.LOCATION_CLASSIFICATIONS['urban']:
            if city in location_clean or location_clean in city:
                return LocationType.URBAN
        
        # Check suburban cities
        for city in self.LOCATION_CLASSIFICATIONS['suburban']:
            if city in location_clean or location_clean in city:
                return LocationType.SUBURBAN
        
        # Try AI classification for unknown locations
        if self.ai_city_classifier and self.ai_city_classifier.enabled:
            try:
                # Parse city name and state/country from location_name
                city_name, state, country = self._parse_location_name(location_name)
                
                logger.debug(f"Using AI to classify unknown location: {location_name}")
                ai_result = self.ai_city_classifier.classify_city(city_name, state, country)
                
                # Convert AI result to LocationType
                location_type = self._convert_ai_density_to_location_type(ai_result.location_density)
                
                # Cache the result for future use
                self.location_intelligence[location_name] = {
                    'type': location_type,
                    'source': 'ai_classification',
                    'confidence': ai_result.confidence,
                    'reasoning': ai_result.reasoning,
                    'response_time_ms': ai_result.response_time_ms,
                    'cache_hit': ai_result.cache_hit,
                    'searches': 0,
                    'total_results': 0,
                    'total_pages': 0,
                    'avg_results_per_page': 0
                }
                
                logger.info(
                    f"🤖 AI classified '{location_name}' as {location_type.value} "
                    f"(confidence: {ai_result.confidence:.2f}, {ai_result.response_time_ms}ms, "
                    f"cache_hit: {ai_result.cache_hit})"
                )
                
                return location_type
                
            except Exception as e:
                logger.warning(f"AI city classification failed for '{location_name}': {str(e)}")
                # Fall through to default classification
        
        # Final fallback for truly unknown locations
        logger.debug(f"No classification found for '{location_name}', defaulting to UNKNOWN")
        return LocationType.UNKNOWN
    
    def should_cluster_locations(self, loc1: Tuple[str, float, float], 
                               loc2: Tuple[str, float, float]) -> bool:
        """
        Determine if two locations should be clustered together.
        
        Args:
            loc1, loc2: Location tuples (name, lat, lon)
            
        Returns:
            True if locations should be clustered
        """
        distance = self.haversine_distance(loc1[1], loc1[2], loc2[1], loc2[2])
        
        # Get location types
        type1 = self.classify_location_type(loc1[0])
        type2 = self.classify_location_type(loc2[0])
        
        # Use the more restrictive threshold
        threshold1 = self.OVERLAP_THRESHOLDS[type1]
        threshold2 = self.OVERLAP_THRESHOLDS[type2]
        threshold = min(threshold1, threshold2)
        
        return distance <= threshold
    
    def cluster_locations(self, locations: List[Tuple[str, float, float]]) -> List[LocationCluster]:
        """
        Cluster nearby locations to minimize overlap and optimize search efficiency.
        
        Args:
            locations: List of (name, latitude, longitude) tuples
            
        Returns:
            List of LocationCluster objects
        """
        if not locations:
            return []
        
        if len(locations) == 1:
            loc = locations[0]
            location_type = self.classify_location_type(loc[0])
            return [LocationCluster(
                centroid_lat=loc[1],
                centroid_lon=loc[2], 
                centroid_name=loc[0],
                locations=[loc],
                radius_miles=0,
                recommended_zoom=self.ZOOM_LEVELS[location_type],
                location_type=location_type
            )]
        
        # Simple clustering algorithm (greedy approach)
        clusters = []
        unprocessed = locations.copy()
        
        while unprocessed:
            # Start new cluster with first unprocessed location
            current_cluster = [unprocessed.pop(0)]
            
            # Find all locations that should be clustered with current cluster
            i = 0
            while i < len(unprocessed):
                should_cluster = False
                
                # Check if this location is close to any location in current cluster
                for cluster_loc in current_cluster:
                    if self.should_cluster_locations(cluster_loc, unprocessed[i]):
                        should_cluster = True
                        break
                
                if should_cluster:
                    current_cluster.append(unprocessed.pop(i))
                else:
                    i += 1
            
            # Create cluster object
            centroid_lat, centroid_lon, centroid_name = self.calculate_centroid(current_cluster)
            
            # Calculate cluster radius
            max_distance = 0
            for loc in current_cluster:
                distance = self.haversine_distance(centroid_lat, centroid_lon, loc[1], loc[2])
                max_distance = max(max_distance, distance)
            
            # Determine cluster type (use most urban type in cluster)
            cluster_types = [self.classify_location_type(loc[0]) for loc in current_cluster]
            
            # Priority order for location types (most urban first)
            type_priority = {
                LocationType.DENSE_URBAN: 4,
                LocationType.URBAN: 3,
                LocationType.SUBURBAN: 2,
                LocationType.RURAL: 1,
                LocationType.UNKNOWN: 0
            }
            
            cluster_type = max(cluster_types, key=lambda t: type_priority[t])
            
            clusters.append(LocationCluster(
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                centroid_name=centroid_name,
                locations=current_cluster,
                radius_miles=max_distance,
                recommended_zoom=self.ZOOM_LEVELS[cluster_type],
                location_type=cluster_type
            ))
        
        logger.info(f"Clustered {len(locations)} locations into {len(clusters)} clusters")
        for i, cluster in enumerate(clusters):
            logger.info(f"Cluster {i+1}: {cluster.location_count} locations, "
                       f"radius {cluster.radius_miles:.1f} miles, "
                       f"zoom {cluster.recommended_zoom}, type {cluster.location_type.value}")
        
        return clusters
    
    def estimate_overlap_percentage(self, cluster1: LocationCluster, cluster2: LocationCluster) -> float:
        """
        Estimate the percentage of overlap between two location clusters.
        
        Args:
            cluster1, cluster2: LocationCluster objects
            
        Returns:
            Estimated overlap percentage (0-100)
        """
        distance = self.haversine_distance(
            cluster1.centroid_lat, cluster1.centroid_lon,
            cluster2.centroid_lat, cluster2.centroid_lon
        )
        
        # Combined search radius (approximate)
        combined_radius = cluster1.radius_miles + cluster2.radius_miles
        
        # Overlap estimation based on distance vs combined radius
        if distance >= combined_radius + 15:  # No overlap
            return 0.0
        elif distance <= max(cluster1.radius_miles, cluster2.radius_miles):  # Significant overlap
            return 60.0
        else:
            # Linear interpolation between significant overlap and no overlap
            overlap_zone = combined_radius + 15 - max(cluster1.radius_miles, cluster2.radius_miles)
            distance_in_zone = distance - max(cluster1.radius_miles, cluster2.radius_miles)
            overlap_percentage = 60.0 * (1 - distance_in_zone / overlap_zone)
            return max(0.0, overlap_percentage)
    
    def update_location_intelligence(self, location_name: str, search_results: Dict):
        """
        Update cached intelligence about a location based on search results.
        
        Args:
            location_name: Name of the location
            search_results: Dictionary with search metrics
        """
        if location_name not in self.location_intelligence:
            self.location_intelligence[location_name] = {
                'searches': 0,
                'total_results': 0,
                'total_pages': 0,
                'avg_results_per_page': 0,
                'type': LocationType.UNKNOWN
            }
        
        intel = self.location_intelligence[location_name]
        # Ensure all required fields exist (for backward compatibility with AI cache)
        if 'searches' not in intel:
            intel['searches'] = 0
        if 'total_results' not in intel:
            intel['total_results'] = 0
        if 'total_pages' not in intel:
            intel['total_pages'] = 0
        if 'avg_results_per_page' not in intel:
            intel['avg_results_per_page'] = 0
            
        intel['searches'] += 1
        intel['total_results'] += search_results.get('total_results', 0)
        intel['total_pages'] += search_results.get('pages_searched', 0)
        
        if intel['total_pages'] > 0:
            intel['avg_results_per_page'] = intel['total_results'] / intel['total_pages']
            
            # Update location type based on results
            if intel['avg_results_per_page'] >= 19:
                intel['type'] = LocationType.DENSE_URBAN
            elif intel['avg_results_per_page'] >= 15:
                intel['type'] = LocationType.URBAN
            elif intel['avg_results_per_page'] >= 10:
                intel['type'] = LocationType.SUBURBAN
            else:
                intel['type'] = LocationType.RURAL
        
        logger.info(f"Updated intelligence for {location_name}: "
                   f"{intel['avg_results_per_page']:.1f} results/page, type {intel['type'].value}")
    
    def get_pagination_rules(self, location_type: LocationType) -> Dict:
        """Get pagination rules for a specific location type."""
        return self.PAGINATION_RULES.get(location_type, self.PAGINATION_RULES[LocationType.UNKNOWN])
    
    def calculate_smart_pagination(self, location_clusters: List[LocationCluster], 
                                 user_max_pages: Optional[int] = None) -> Dict:
        """
        Calculate intelligent pagination strategy for all clusters.
        
        Args:
            location_clusters: List of location clusters
            user_max_pages: Optional user override for max pages
            
        Returns:
            Dictionary with pagination strategy for each cluster
        """
        pagination_strategy = {}
        
        for i, cluster in enumerate(location_clusters):
            rules = self.get_pagination_rules(cluster.location_type)
            
            # Calculate recommended pages for this cluster
            if user_max_pages is not None:
                # User override: respect their choice but cap at location type max
                recommended_max = min(user_max_pages, rules['max_pages'])
            else:
                # Auto-pagination: use location type defaults
                recommended_max = rules['max_pages']
            
            # Ensure minimum pages are respected
            recommended_min = rules['min_pages']
            
            pagination_strategy[f"cluster_{i}"] = {
                'cluster_name': cluster.centroid_name,
                'location_type': cluster.location_type.value,
                'min_pages': recommended_min,
                'max_pages': recommended_max,
                'stop_threshold': rules['stop_threshold'],
                'duplicate_threshold': rules['duplicate_threshold'],
                'description': rules['description'],
                'auto_pagination': user_max_pages is None
            }
        
        return pagination_strategy
    
    def _parse_location_name(self, location_name: str) -> Tuple[str, str, str]:
        """
        Parse location name into city, state, and country components.
        
        Args:
            location_name: Full location string (e.g., "Austin, TX", "London, UK")
            
        Returns:
            Tuple of (city_name, state, country)
        """
        parts = [part.strip() for part in location_name.split(',')]
        
        city_name = parts[0] if parts else location_name
        state = ""
        country = ""
        
        if len(parts) >= 2:
            # Check if second part looks like a US state (2 letters) or country
            second_part = parts[1].upper()
            if len(second_part) == 2 and second_part.isalpha():
                # Likely a US state
                state = second_part
                country = "USA"
            else:
                # Likely a country or longer state name
                country = parts[1]
                
        if len(parts) >= 3:
            # Third part is likely country
            country = parts[2]
        
        # Handle common international patterns
        if country.upper() in ["UK", "UNITED KINGDOM"]:
            country = "United Kingdom"
        elif country.upper() in ["USA", "US", "UNITED STATES"]:
            country = "USA"
        
        return city_name, state, country
    
    def _convert_ai_density_to_location_type(self, ai_density) -> LocationType:
        """
        Convert AI city classifier density enum to GeographicOptimizer LocationType.
        
        Args:
            ai_density: LocationDensity enum from city classifier
            
        Returns:
            LocationType enum value
        """
        # Import here to avoid circular imports
        from services.city_classifier import LocationDensity
        
        density_mapping = {
            LocationDensity.DENSE_URBAN: LocationType.DENSE_URBAN,
            LocationDensity.URBAN: LocationType.URBAN,
            LocationDensity.SUBURBAN: LocationType.SUBURBAN,
            LocationDensity.RURAL: LocationType.RURAL
        }
        
        return density_mapping.get(ai_density, LocationType.UNKNOWN)