import hashlib
import json
import logging
import time
from typing import Optional, Dict, List, Tuple
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

class LocationDensity(Enum):
    """Business density classification for cities."""
    DENSE_URBAN = "dense_urban"
    URBAN = "urban" 
    SUBURBAN = "suburban"
    RURAL = "rural"

@dataclass
class CityClassificationResult:
    """Result of AI city classification with metadata."""
    city_name: str
    location_density: LocationDensity
    confidence: float
    reasoning: str
    cache_hit: bool
    response_time_ms: int
    population_estimate: Optional[int] = None
    country: Optional[str] = None

class AICityClassifier:
    """
    AI-powered city classification using Groq's fast inference.
    
    Classifies cities into business density categories (dense_urban, urban, suburban, rural)
    using LLM knowledge of global geography, population, and economic patterns.
    """
    
    # Example cities for each category to help AI understand patterns
    CATEGORY_EXAMPLES = {
        LocationDensity.DENSE_URBAN: [
            "New York City, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
            "Austin, TX", "San Francisco, CA", "Seattle, WA", "Boston, MA", 
            "Miami, FL", "Denver, CO", "London, UK", "Tokyo, Japan", "Paris, France"
        ],
        LocationDensity.URBAN: [
            "Boise, ID", "Portland, OR", "Reno, NV", "Birmingham, AL",
            "Nashville, TN", "Sacramento, CA", "Orlando, FL", "Cincinnati, OH",
            "Vancouver, Canada", "Edinburgh, Scotland", "Brisbane, Australia"
        ],
        LocationDensity.SUBURBAN: [
            "Frisco, TX", "Round Rock, TX", "Plano, TX", "Sugar Land, TX",
            "Pearland, TX", "Allen, TX", "Cedar Park, TX", "Georgetown, TX",
            "Flower Mound, TX", "Katy, TX", "Naperville, IL", "Irvine, CA"
        ],
        LocationDensity.RURAL: [
            "Small towns under 50,000 population", "Agricultural communities",
            "Mountain towns", "Coastal villages", "Desert communities"
        ]
    }
    
    def __init__(self, cache_manager=None):
        """
        Initialize AI city classifier.
        
        Args:
            cache_manager: Optional cache manager for storing classification results
        """
        self.cache_manager = cache_manager
        self.groq_client = None
        self.enabled = False
        
        # Initialize Groq client if available and configured
        if GROQ_AVAILABLE and settings.GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                self.enabled = True
                logger.info("🌍 AI city classification enabled with Groq")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client for city classification: {str(e)}")
        else:
            if not GROQ_AVAILABLE:
                logger.warning("Groq library not available for city classification")
            elif not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not configured. City classification disabled.")
    
    def classify_city(self, city_name: str, state: str = "", country: str = "") -> CityClassificationResult:
        """
        Classify a city into business density categories using AI.
        
        Args:
            city_name: Name of the city (e.g., "Austin", "Boise")
            state: State/province if applicable (e.g., "TX", "ID")
            country: Country if applicable (e.g., "USA", "Canada")
            
        Returns:
            CityClassificationResult with density classification and metadata
        """
        start_time = time.time()
        full_city_name = self._format_city_name(city_name, state, country)
        
        # Check cache first
        cache_key = self._generate_cache_key(full_city_name)
        
        if self.cache_manager:
            cached_result = self.cache_manager.get_city_classification(cache_key)
            if cached_result is not None:
                response_time = int((time.time() - start_time) * 1000)
                
                # Parse cached result
                density, confidence, reasoning, population, country_parsed = cached_result
                
                return CityClassificationResult(
                    city_name=full_city_name,
                    location_density=LocationDensity(density),
                    confidence=confidence,
                    reasoning=reasoning,
                    cache_hit=True,
                    response_time_ms=response_time,
                    population_estimate=population,
                    country=country_parsed
                )
        
        # Fallback to conservative classification if AI not available
        if not self.enabled:
            density = self._fallback_classification(city_name, state, country)
            response_time = int((time.time() - start_time) * 1000)
            
            result = CityClassificationResult(
                city_name=full_city_name,
                location_density=density,
                confidence=0.6,  # Lower confidence for fallback
                reasoning="Conservative fallback (AI unavailable)",
                cache_hit=False,
                response_time_ms=response_time
            )
            
            # Cache fallback result
            if self.cache_manager:
                self.cache_manager.set_city_classification(
                    cache_key, full_city_name, density.value, 0.6, result.reasoning, None, country
                )
            
            return result
        
        # Use AI classification
        try:
            ai_result = self._ask_ai(full_city_name)
            response_time = int((time.time() - start_time) * 1000)
            
            # Cache the result with long TTL (cities don't change density often)
            if self.cache_manager:
                self.cache_manager.set_city_classification(
                    cache_key, 
                    full_city_name,
                    ai_result['density'].value,
                    ai_result['confidence'],
                    ai_result['reasoning'],
                    ai_result.get('population'),
                    ai_result.get('country')
                )
            
            return CityClassificationResult(
                city_name=full_city_name,
                location_density=ai_result['density'],
                confidence=ai_result['confidence'],
                reasoning=ai_result['reasoning'],
                cache_hit=False,
                response_time_ms=response_time,
                population_estimate=ai_result.get('population'),
                country=ai_result.get('country')
            )
            
        except Exception as e:
            logger.error(f"AI city classification failed: {str(e)}")
            
            # Fallback to conservative classification
            density = self._fallback_classification(city_name, state, country)
            response_time = int((time.time() - start_time) * 1000)
            
            return CityClassificationResult(
                city_name=full_city_name,
                location_density=density,
                confidence=0.4,  # Lower confidence due to error
                reasoning=f"Fallback due to AI error: {str(e)}",
                cache_hit=False,
                response_time_ms=response_time
            )
    
    def classify_batch(self, cities: List[Tuple[str, str, str]]) -> List[CityClassificationResult]:
        """
        Classify multiple cities in a batch for efficiency.
        
        Args:
            cities: List of (city_name, state, country) tuples
            
        Returns:
            List of CityClassificationResult objects
        """
        results = []
        
        for city_name, state, country in cities:
            result = self.classify_city(city_name, state, country)
            results.append(result)
            
            # Small delay between API calls to be respectful
            if not result.cache_hit:
                time.sleep(0.1)
        
        return results
    
    def _ask_ai(self, full_city_name: str) -> Dict:
        """Ask AI to classify a city's business density."""
        prompt = self._create_classification_prompt(full_city_name)
        
        try:
            response = self.groq_client.chat.completions.create(
                model=settings.AI_VALIDATION_MODEL,  # Use same model as validation
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a global geography and urban planning expert. "
                            "Classify cities into business density categories based on "
                            "population, economic activity, and urban development patterns. "
                            "Respond with ONLY the category name: dense_urban, urban, suburban, or rural."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent classifications
                max_tokens=50,    # More tokens than validation since we want reasoning
                top_p=0.9
            )
            
            ai_response = response.choices[0].message.content.strip().lower()
            
            # Parse the response to extract density category
            density = self._parse_ai_response(ai_response)
            
            logger.debug(f"AI city classification: '{full_city_name}' -> {density.value}")
            
            return {
                'density': density,
                'confidence': 0.9,
                'reasoning': f"AI classification: {ai_response}",
                'population': None,  # Could be enhanced to extract population estimates
                'country': None      # Could be enhanced to extract country info
            }
            
        except Exception as e:
            logger.error(f"Groq API error in city classification: {str(e)}")
            raise
    
    def _create_classification_prompt(self, full_city_name: str) -> str:
        """Create a classification prompt for the AI."""
        
        examples_text = ""
        for density, examples in self.CATEGORY_EXAMPLES.items():
            if density != LocationDensity.RURAL:  # Rural examples are descriptive
                examples_text += f"\n{density.value}: {', '.join(examples[:5])}..."
        
        prompt = f"""Classify this city into ONE business density category:

CATEGORIES:
• dense_urban: Major metropolitan centers, tech hubs, financial districts
• urban: Mid-size cities, state capitals, regional business centers  
• suburban: Planned communities, suburbs of major cities, smaller cities
• rural: Small towns under 50,000 population, agricultural areas

EXAMPLES:{examples_text}

CITY TO CLASSIFY: "{full_city_name}"

Consider: population size, economic activity, business concentration, urban development

Classification:"""
        
        return prompt
    
    def _parse_ai_response(self, ai_response: str) -> LocationDensity:
        """Parse AI response to extract LocationDensity enum."""
        ai_response = ai_response.lower().strip()
        
        # Direct matches
        if "dense_urban" in ai_response or "dense urban" in ai_response:
            return LocationDensity.DENSE_URBAN
        elif "suburban" in ai_response:
            return LocationDensity.SUBURBAN
        elif "rural" in ai_response:
            return LocationDensity.RURAL
        elif "urban" in ai_response:
            return LocationDensity.URBAN
        
        # Fallback parsing
        if any(word in ai_response for word in ["major", "metropolitan", "tech hub", "financial"]):
            return LocationDensity.DENSE_URBAN
        elif any(word in ai_response for word in ["suburb", "planned", "residential"]):
            return LocationDensity.SUBURBAN
        elif any(word in ai_response for word in ["small", "town", "rural", "agricultural"]):
            return LocationDensity.RURAL
        else:
            # Default to urban if unclear
            return LocationDensity.URBAN
    
    def _fallback_classification(self, city_name: str, state: str = "", country: str = "") -> LocationDensity:
        """
        Conservative fallback classification when AI is not available.
        
        Uses simple heuristics based on known patterns.
        """
        city_lower = city_name.lower()
        
        # Known major US cities (basic list)
        major_cities = {
            "new york", "los angeles", "chicago", "houston", "austin", 
            "san francisco", "seattle", "boston", "miami", "denver",
            "atlanta", "philadelphia", "phoenix", "san diego", "dallas"
        }
        
        # Known suburban patterns
        suburban_indicators = ["frisco", "plano", "round rock", "sugar land", "pearland"]
        
        if any(major in city_lower for major in major_cities):
            return LocationDensity.DENSE_URBAN
        elif any(indicator in city_lower for indicator in suburban_indicators):
            return LocationDensity.SUBURBAN
        elif state.lower() in ["tx", "ca", "ny", "fl"] and country.lower() in ["", "usa", "us"]:
            return LocationDensity.URBAN  # Assume urban for major states
        else:
            return LocationDensity.SUBURBAN  # Conservative default
    
    def _format_city_name(self, city_name: str, state: str = "", country: str = "") -> str:
        """Format city name for consistent processing."""
        parts = [city_name.strip()]
        
        if state.strip():
            parts.append(state.strip())
        
        if country.strip() and country.lower() not in ["usa", "us", "united states"]:
            parts.append(country.strip())
        
        return ", ".join(parts)
    
    def _generate_cache_key(self, full_city_name: str) -> str:
        """Generate a cache key for the city classification request."""
        # Normalize the city name for consistent caching
        normalized = full_city_name.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_stats(self) -> Dict:
        """Get city classification statistics."""
        stats = {
            'enabled': self.enabled,
            'groq_available': GROQ_AVAILABLE,
            'has_api_key': bool(settings.GROQ_API_KEY),
            'cache_enabled': self.cache_manager is not None,
            'category_examples': {
                density.value: len(examples) for density, examples in self.CATEGORY_EXAMPLES.items()
            }
        }
        
        if self.cache_manager:
            cache_stats = self.cache_manager.get_city_cache_stats()
            stats.update(cache_stats)
        
        return stats