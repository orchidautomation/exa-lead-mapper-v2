"""
AI-powered location expansion service.

This service intelligently expands broad location terms (like counties or regions)
into specific cities and towns, considering population density, business activity,
and search intent.
"""

import hashlib
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

from config.settings import settings

logger = logging.getLogger(__name__)

class LocationType(Enum):
    """Types of location inputs."""
    CITY = "city"
    COUNTY = "county"
    REGION = "region"
    STATE = "state"
    ZIP = "zip"
    METRO_AREA = "metro_area"

@dataclass
class LocationSuggestion:
    """A suggested location with metadata."""
    location: str
    location_type: LocationType
    population_estimate: Optional[int]
    business_density: str  # high, medium, low
    relevance_score: float  # 0.0 to 1.0
    reasoning: str

@dataclass
class LocationExpansionResult:
    """Complete location expansion result."""
    original_location: str
    detected_type: LocationType
    suggested_locations: List[LocationSuggestion]
    selected_locations: List[str]
    expansion_reasoning: str
    coverage_analysis: Dict
    cache_hit: bool
    response_time_ms: int
    ai_enabled: bool

class LocationExpander:
    """
    Intelligently expands broad location terms into specific searchable locations.
    """
    
    # Knowledge base for common county/region expansions
    LOCATION_EXPANSIONS = {
        'knox county': {
            'state': 'TN',
            'cities': [
                ('Knoxville', 'city', 190740, 'high'),
                ('Farragut', 'town', 23506, 'medium'),
                ('Powell', 'cdp', 13802, 'medium'),
                ('Karns', 'cdp', 11000, 'medium'),
                ('Halls Crossroads', 'cdp', 10000, 'low'),
                ('Mascot', 'cdp', 2700, 'low'),
                ('Corryton', 'cdp', 2000, 'low')
            ]
        },
        'davidson county': {
            'state': 'TN',
            'cities': [
                ('Nashville', 'city', 689447, 'high'),
                ('Belle Meade', 'city', 2912, 'medium'),
                ('Forest Hills', 'city', 4889, 'medium'),
                ('Oak Hill', 'city', 4529, 'medium'),
                ('Berry Hill', 'city', 464, 'medium')
            ]
        },
        'shelby county': {
            'state': 'TN',
            'cities': [
                ('Memphis', 'city', 633104, 'high'),
                ('Bartlett', 'city', 54613, 'medium'),
                ('Collierville', 'town', 51324, 'medium'),
                ('Germantown', 'city', 39038, 'medium'),
                ('Millington', 'city', 10176, 'low'),
                ('Arlington', 'town', 14001, 'low')
            ]
        }
    }
    
    def __init__(self, geocoding_service=None, cache_manager=None):
        """
        Initialize location expander.
        
        Args:
            geocoding_service: Service for geocoding locations
            cache_manager: Optional cache manager for storing expansion results
        """
        self.geocoding_service = geocoding_service
        self.cache_manager = cache_manager
        self.groq_client = None
        self.enabled = False
        
        # Initialize Groq client if available
        if GROQ_AVAILABLE and settings.GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                self.enabled = True
                logger.info("📍 AI-powered location expansion enabled with Groq")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq for location expansion: {str(e)}")
        else:
            if not GROQ_AVAILABLE:
                logger.warning("Groq library not available for location expansion")
            elif not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not configured. Location expansion limited.")
    
    def expand_location(self,
                       location: str,
                       business_verticals: List[str] = None,
                       search_intent: Optional[str] = None,
                       max_locations: int = 5,
                       include_suburbs: bool = True) -> LocationExpansionResult:
        """
        Expand a broad location into specific searchable locations.
        
        Args:
            location: The location to expand (e.g., "Knox County, TN")
            business_verticals: Types of businesses being searched (affects city selection)
            search_intent: Context about the search (e.g., "affluent areas for luxury services")
            max_locations: Maximum number of locations to return
            include_suburbs: Whether to include suburbs and smaller areas
            
        Returns:
            LocationExpansionResult with selected locations and analysis
        """
        start_time = time.time()
        
        # Normalize location
        location_clean = location.strip().lower()
        
        # Detect location type
        location_type = self._detect_location_type(location_clean)
        
        # Check if expansion is needed
        if location_type in [LocationType.CITY, LocationType.ZIP]:
            return self._create_no_expansion_result(location, location_type, start_time)
        
        # Check cache
        cache_key = self._generate_cache_key(location_clean, business_verticals, search_intent)
        if self.cache_manager:
            cached_result = self._get_cached_expansion(cache_key)
            if cached_result:
                response_time = int((time.time() - start_time) * 1000)
                cached_result.response_time_ms = response_time
                cached_result.cache_hit = True
                return cached_result
        
        # Get expansion suggestions
        if self.enabled and location_type in [LocationType.COUNTY, LocationType.REGION, LocationType.METRO_AREA]:
            suggestions = self._get_ai_suggestions(location_clean, location_type, business_verticals, search_intent)
        else:
            suggestions = self._get_fallback_suggestions(location_clean, location_type)
        
        # Select optimal locations
        selected_locations = self._select_optimal_locations(
            suggestions, max_locations, include_suburbs, business_verticals
        )
        
        # Generate analysis
        coverage_analysis = self._analyze_coverage(location, selected_locations, suggestions)
        
        response_time = int((time.time() - start_time) * 1000)
        
        result = LocationExpansionResult(
            original_location=location,
            detected_type=location_type,
            suggested_locations=suggestions,
            selected_locations=selected_locations,
            expansion_reasoning=self._generate_expansion_reasoning(
                location, location_type, selected_locations, business_verticals
            ),
            coverage_analysis=coverage_analysis,
            cache_hit=False,
            response_time_ms=response_time,
            ai_enabled=self.enabled
        )
        
        # Cache the result
        if self.cache_manager:
            self._cache_expansion(cache_key, result)
        
        return result
    
    def _detect_location_type(self, location: str) -> LocationType:
        """Detect the type of location from the input string."""
        location_lower = location.lower()
        
        # Check for ZIP code
        import re
        if re.match(r'^\d{5}(-\d{4})?$', location.replace(' ', '')):
            return LocationType.ZIP
        
        # Check for county
        if 'county' in location_lower:
            return LocationType.COUNTY
        
        # Check for metro area indicators
        if any(term in location_lower for term in ['metro', 'metropolitan', 'greater', 'area']):
            return LocationType.METRO_AREA
        
        # Check for region indicators
        if any(term in location_lower for term in ['region', 'valley', 'coast', 'bay area']):
            return LocationType.REGION
        
        # Check for state (just state abbreviation or full state name)
        state_pattern = r'^[a-z]{2}$|^[a-z]+\s*,?\s*[a-z]{2}$'
        if re.match(state_pattern, location_lower):
            parts = location_lower.split(',')
            if len(parts) == 1 or (len(parts) == 2 and len(parts[1].strip()) == 2):
                return LocationType.STATE
        
        # Default to city
        return LocationType.CITY
    
    def _get_ai_suggestions(self, location: str, location_type: LocationType,
                           business_verticals: List[str] = None,
                           search_intent: Optional[str] = None) -> List[LocationSuggestion]:
        """Get AI-powered location expansion suggestions."""
        try:
            prompt = self._create_expansion_prompt(location, location_type, business_verticals, search_intent)
            
            response = self.groq_client.chat.completions.create(
                model=settings.AI_VALIDATION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a geographic expert who knows cities, towns, and neighborhoods. "
                            "Suggest specific locations within the given area that would be relevant "
                            "for business searches. Focus on real, searchable place names."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content.strip()
            logger.debug(f"AI location expansion response: {ai_response}")
            return self._parse_ai_suggestions(ai_response, location)
            
        except Exception as e:
            logger.error(f"AI location expansion failed: {str(e)}")
            return self._get_fallback_suggestions(location, location_type)
    
    def _create_expansion_prompt(self, location: str, location_type: LocationType,
                                business_verticals: List[str] = None,
                                search_intent: Optional[str] = None) -> str:
        """Create AI prompt for location expansion."""
        business_context = ""
        if business_verticals:
            business_context = f"Business types: {', '.join(business_verticals)}"
        
        intent_context = ""
        if search_intent:
            intent_context = f"Search intent: {search_intent}"
        
        prompt = f"""Suggest cities/towns within "{location}" for business searches.

Location type: {location_type.value}
{business_context}
{intent_context}

Provide 5-7 specific city/town names that exist in this area.
Consider population size, business activity, and relevance.

Format each suggestion as:
LOCATION: City/Town Name, State
TYPE: city/town/cdp
POPULATION: estimated population (number only)
DENSITY: high/medium/low (business density)
RELEVANCE: 0.1-1.0
REASON: why this location is relevant

Example:
LOCATION: Knoxville, TN
TYPE: city
POPULATION: 190740
DENSITY: high
RELEVANCE: 0.9
REASON: Largest city in the county with high business concentration"""

        return prompt
    
    def _parse_ai_suggestions(self, ai_response: str, original_location: str) -> List[LocationSuggestion]:
        """Parse AI response into LocationSuggestion objects."""
        suggestions = []
        lines = ai_response.strip().split('\n')
        
        current_suggestion = {}
        
        for line in lines:
            line = line.strip()
            if 'LOCATION:' in line:
                if current_suggestion and current_suggestion.get('location'):
                    suggestions.append(self._create_suggestion_from_dict(current_suggestion))
                loc_start = line.find('LOCATION:') + 9
                current_suggestion = {'location': line[loc_start:].strip()}
            elif 'TYPE:' in line:
                type_start = line.find('TYPE:') + 5
                type_str = line[type_start:].strip()
                current_suggestion['type'] = self._parse_location_type(type_str)
            elif 'POPULATION:' in line:
                try:
                    pop_start = line.find('POPULATION:') + 11
                    current_suggestion['population'] = int(line[pop_start:].strip().replace(',', ''))
                except:
                    current_suggestion['population'] = None
            elif 'DENSITY:' in line:
                density_start = line.find('DENSITY:') + 8
                current_suggestion['density'] = line[density_start:].strip()
            elif 'RELEVANCE:' in line:
                try:
                    rel_start = line.find('RELEVANCE:') + 10
                    current_suggestion['relevance'] = float(line[rel_start:].strip())
                except:
                    current_suggestion['relevance'] = 0.5
            elif 'REASON:' in line:
                reason_start = line.find('REASON:') + 7
                current_suggestion['reasoning'] = line[reason_start:].strip()
        
        # Add last suggestion
        if current_suggestion and current_suggestion.get('location'):
            suggestions.append(self._create_suggestion_from_dict(current_suggestion))
        
        # Fallback if parsing failed
        if not suggestions:
            logger.warning(f"AI location parsing failed for '{original_location}', using fallback")
            return self._get_fallback_suggestions(original_location, LocationType.COUNTY)
        
        return suggestions
    
    def _create_suggestion_from_dict(self, suggestion_dict: Dict) -> LocationSuggestion:
        """Create LocationSuggestion from parsed dictionary."""
        return LocationSuggestion(
            location=suggestion_dict.get('location', ''),
            location_type=suggestion_dict.get('type', LocationType.CITY),
            population_estimate=suggestion_dict.get('population'),
            business_density=suggestion_dict.get('density', 'medium'),
            relevance_score=suggestion_dict.get('relevance', 0.5),
            reasoning=suggestion_dict.get('reasoning', 'Suggested location')
        )
    
    def _parse_location_type(self, type_str: str) -> LocationType:
        """Parse location type from string."""
        type_map = {
            'city': LocationType.CITY,
            'town': LocationType.CITY,
            'cdp': LocationType.CITY,  # Census Designated Place
            'county': LocationType.COUNTY,
            'region': LocationType.REGION,
            'metro': LocationType.METRO_AREA
        }
        return type_map.get(type_str.lower(), LocationType.CITY)
    
    def _get_fallback_suggestions(self, location: str, location_type: LocationType) -> List[LocationSuggestion]:
        """Get fallback suggestions using knowledge base."""
        location_key = location.lower().replace(',', '').replace('tn', '').strip()
        
        # Check knowledge base
        if location_key in self.LOCATION_EXPANSIONS:
            data = self.LOCATION_EXPANSIONS[location_key]
            state = data['state']
            suggestions = []
            
            for city_data in data['cities']:
                city_name, city_type, population, density = city_data
                suggestions.append(LocationSuggestion(
                    location=f"{city_name}, {state}",
                    location_type=LocationType.CITY,
                    population_estimate=population,
                    business_density=density,
                    relevance_score=self._calculate_relevance(population, density),
                    reasoning=f"Known {city_type} in {location}"
                ))
            
            return suggestions
        
        # Generic fallback
        return [LocationSuggestion(
            location=location,
            location_type=LocationType.CITY,
            population_estimate=None,
            business_density='medium',
            relevance_score=0.5,
            reasoning="No expansion available"
        )]
    
    def _calculate_relevance(self, population: Optional[int], density: str) -> float:
        """Calculate relevance score based on population and density."""
        density_scores = {'high': 0.3, 'medium': 0.2, 'low': 0.1}
        density_score = density_scores.get(density, 0.2)
        
        if population:
            if population > 100000:
                pop_score = 0.6
            elif population > 25000:
                pop_score = 0.4
            elif population > 10000:
                pop_score = 0.3
            else:
                pop_score = 0.2
        else:
            pop_score = 0.3
        
        return min(0.9, pop_score + density_score)
    
    def _select_optimal_locations(self, suggestions: List[LocationSuggestion],
                                 max_locations: int, include_suburbs: bool,
                                 business_verticals: List[str] = None) -> List[str]:
        """Select optimal locations from suggestions."""
        if not suggestions:
            return []
        
        # Filter based on include_suburbs
        if not include_suburbs:
            # Only include high-density areas or cities with population > 25000
            filtered = [s for s in suggestions 
                       if s.business_density == 'high' or 
                       (s.population_estimate and s.population_estimate > 25000)]
            if filtered:
                suggestions = filtered
        
        # Sort by relevance score
        sorted_suggestions = sorted(suggestions, key=lambda s: s.relevance_score, reverse=True)
        
        # Select top locations
        selected = []
        for suggestion in sorted_suggestions[:max_locations]:
            selected.append(suggestion.location)
        
        return selected
    
    def _analyze_coverage(self, original: str, selected: List[str], 
                         all_suggestions: List[LocationSuggestion]) -> Dict:
        """Analyze geographic coverage of selected locations."""
        if not selected:
            return {"coverage": "none", "reasoning": "No expansion performed"}
        
        total_population = sum(s.population_estimate for s in all_suggestions 
                              if s.population_estimate and s.location in selected)
        
        high_density_count = sum(1 for s in all_suggestions 
                                if s.location in selected and s.business_density == 'high')
        
        return {
            "original_location": original,
            "expanded_locations": len(selected),
            "estimated_population_coverage": total_population,
            "high_density_locations": high_density_count,
            "coverage_quality": "good" if high_density_count > 0 else "moderate"
        }
    
    def _generate_expansion_reasoning(self, location: str, location_type: LocationType,
                                     selected: List[str], business_verticals: List[str] = None) -> str:
        """Generate human-readable reasoning for expansion."""
        if len(selected) <= 1:
            return f"No expansion needed for {location_type.value} '{location}'"
        
        vertical_context = ""
        if business_verticals:
            vertical_context = f" for {', '.join(business_verticals[:2])} businesses"
        
        return (f"Expanded {location_type.value} '{location}' to {len(selected)} "
                f"cities/towns{vertical_context} to ensure comprehensive coverage")
    
    def _create_no_expansion_result(self, location: str, location_type: LocationType, 
                                   start_time: float) -> LocationExpansionResult:
        """Create result when no expansion is needed."""
        response_time = int((time.time() - start_time) * 1000)
        
        return LocationExpansionResult(
            original_location=location,
            detected_type=location_type,
            suggested_locations=[],
            selected_locations=[location],
            expansion_reasoning=f"'{location}' is already a specific {location_type.value}, no expansion needed",
            coverage_analysis={
                "coverage": "single_location",
                "no_expansion": True
            },
            cache_hit=False,
            response_time_ms=response_time,
            ai_enabled=self.enabled
        )
    
    def _generate_cache_key(self, location: str, verticals: List[str] = None, 
                           intent: Optional[str] = None) -> str:
        """Generate cache key for expansion request."""
        cache_data = {
            'location': location,
            'verticals': sorted(verticals) if verticals else [],
            'intent': intent or ''
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_expansion(self, cache_key: str) -> Optional[LocationExpansionResult]:
        """Get cached expansion result."""
        # Placeholder for cache implementation
        return None
    
    def _cache_expansion(self, cache_key: str, result: LocationExpansionResult):
        """Cache expansion result."""
        # Placeholder for cache implementation
        pass
    
    def get_stats(self) -> Dict:
        """Get expansion service statistics."""
        return {
            'enabled': self.enabled,
            'groq_available': GROQ_AVAILABLE,
            'has_api_key': bool(settings.GROQ_API_KEY),
            'knowledge_base_counties': len(self.LOCATION_EXPANSIONS),
            'cache_enabled': self.cache_manager is not None
        }