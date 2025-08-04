import asyncio
import hashlib
import json
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import time

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of AI validation with metadata."""
    is_match: bool
    confidence: float
    reasoning: str
    cache_hit: bool
    response_time_ms: int

class AIVerticalValidator:
    """
    AI-powered vertical validation using Groq's fast inference.
    
    Uses Llama 3.1 8B to intelligently determine if business types
    match the searched vertical, with aggressive caching for cost efficiency.
    """
    
    def __init__(self, cache_manager=None):
        """
        Initialize AI validator.
        
        Args:
            cache_manager: Optional cache manager for storing validation results
        """
        self.cache_manager = cache_manager
        self.groq_client = None
        self.enabled = False
        self.batch_validator = None
        
        # Initialize Groq client if available and configured
        if GROQ_AVAILABLE and settings.GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                self.enabled = True
                logger.info("🤖 AI validation enabled with Groq")
                
                # Initialize batch validator
                from services.batch_ai_validator import BatchAIValidator
                self.batch_validator = BatchAIValidator(
                    groq_client=self.groq_client,
                    cache_manager=cache_manager,
                    batch_size=15
                )
                logger.info("🚀 Batch AI validation initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {str(e)}")
        else:
            if not GROQ_AVAILABLE:
                logger.warning("Groq library not available. Install with: pip install groq")
            elif not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not configured. AI validation disabled.")
    
    def validate_match(self, vertical: str, business_types: List[str], 
                      business_title: str = "") -> ValidationResult:
        """
        Validate if business types match the searched vertical using AI.
        
        Args:
            vertical: The search term (e.g., "coffee shop")
            business_types: List of business types from Serper (e.g., ["Coffee shop", "Cafe"])
            business_title: Optional business title for additional context
            
        Returns:
            ValidationResult with match decision and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(vertical, business_types, business_title)
        
        if self.cache_manager:
            cached_result = self.cache_manager.get_validation(cache_key)
            if cached_result is not None:
                response_time = int((time.time() - start_time) * 1000)
                return ValidationResult(
                    is_match=cached_result,
                    confidence=1.0,  # Cached results are trusted
                    reasoning="Cached result",
                    cache_hit=True,
                    response_time_ms=response_time
                )
        
        # Fallback to rule-based validation if AI not available
        if not self.enabled:
            is_match = self._fallback_validation(vertical, business_types, business_title)
            response_time = int((time.time() - start_time) * 1000)
            
            result = ValidationResult(
                is_match=is_match,
                confidence=0.7,  # Lower confidence for fallback
                reasoning="Rule-based fallback (AI unavailable)",
                cache_hit=False,
                response_time_ms=response_time
            )
            
            # Cache fallback result
            if self.cache_manager:
                self.cache_manager.set_validation(cache_key, is_match)
            
            return result
        
        # Use AI validation
        try:
            ai_result = self._ask_ai(vertical, business_types, business_title)
            response_time = int((time.time() - start_time) * 1000)
            
            # Cache the result
            if self.cache_manager:
                self.cache_manager.set_validation(cache_key, ai_result['is_match'])
            
            return ValidationResult(
                is_match=ai_result['is_match'],
                confidence=ai_result.get('confidence', 0.9),
                reasoning=ai_result.get('reasoning', 'AI decision'),
                cache_hit=False,
                response_time_ms=response_time
            )
            
        except Exception as e:
            logger.error(f"AI validation failed: {str(e)}")
            
            # Fallback to rule-based validation
            is_match = self._fallback_validation(vertical, business_types, business_title)
            response_time = int((time.time() - start_time) * 1000)
            
            return ValidationResult(
                is_match=is_match,
                confidence=0.5,  # Lower confidence due to error
                reasoning=f"Fallback due to AI error: {str(e)}",
                cache_hit=False,
                response_time_ms=response_time
            )
    
    def validate_batch(self, validations: List[Tuple[str, List[str], str]]) -> List[ValidationResult]:
        """
        Validate multiple vertical-business type combinations in a single request.
        
        Args:
            validations: List of (vertical, business_types, business_title) tuples
            
        Returns:
            List of ValidationResult objects
        """
        # Use batch validator if available
        if self.batch_validator and self.enabled:
            try:
                return self.batch_validator.validate_batch(validations)
            except Exception as e:
                logger.error(f"Batch validator failed, falling back to sequential: {str(e)}")
        
        # Fallback to sequential processing
        results = []
        
        # Check cache for all items first
        uncached_items = []
        for i, (vertical, business_types, business_title) in enumerate(validations):
            cache_key = self._generate_cache_key(vertical, business_types, business_title)
            
            if self.cache_manager:
                cached_result = self.cache_manager.get_validation(cache_key)
                if cached_result is not None:
                    results.append((i, ValidationResult(
                        is_match=cached_result,
                        confidence=1.0,
                        reasoning="Cached result",
                        cache_hit=True,
                        response_time_ms=1
                    )))
                    continue
            
            uncached_items.append((i, vertical, business_types, business_title))
        
        # Process uncached items
        if uncached_items and self.enabled:
            try:
                batch_results = self._ask_ai_batch(uncached_items)
                for (idx, result) in batch_results:
                    results.append((idx, result))
            except Exception as e:
                logger.error(f"Batch AI validation failed: {str(e)}")
                # Fallback for all uncached items
                for (idx, vertical, business_types, business_title) in uncached_items:
                    is_match = self._fallback_validation(vertical, business_types, business_title)
                    results.append((idx, ValidationResult(
                        is_match=is_match,
                        confidence=0.5,
                        reasoning=f"Batch fallback: {str(e)}",
                        cache_hit=False,
                        response_time_ms=10
                    )))
        elif uncached_items:
            # Use fallback for all uncached items
            for (idx, vertical, business_types, business_title) in uncached_items:
                is_match = self._fallback_validation(vertical, business_types, business_title)
                results.append((idx, ValidationResult(
                    is_match=is_match,
                    confidence=0.7,
                    reasoning="Rule-based fallback",
                    cache_hit=False,
                    response_time_ms=5
                )))
        
        # Sort results by original index and return just the ValidationResult objects
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def _ask_ai(self, vertical: str, business_types: List[str], business_title: str = "") -> Dict:
        """Ask AI to validate a single vertical-business type match."""
        prompt = self._create_validation_prompt(vertical, business_types, business_title)
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a business categorization expert. Determine if a business "
                            "matches a search query based on its types and title. "
                            "Respond with only 'YES' or 'NO'."
                        )
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=10,
                top_p=0.9
            )
            
            ai_response = response.choices[0].message.content.strip().upper()
            is_match = ai_response.startswith('YES')
            
            logger.debug(f"AI validation: '{vertical}' vs {business_types} -> {ai_response}")
            
            return {
                'is_match': is_match,
                'confidence': 0.9,
                'reasoning': f"AI decision: {ai_response}"
            }
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise
    
    def _ask_ai_batch(self, validations: List[Tuple[int, str, List[str], str]]) -> List[Tuple[int, ValidationResult]]:
        """Ask AI to validate multiple items in a single request."""
        # For now, process individually (Groq doesn't have native batching)
        # Could be optimized with a single prompt containing multiple validations
        results = []
        
        for idx, vertical, business_types, business_title in validations:
            try:
                ai_result = self._ask_ai(vertical, business_types, business_title)
                result = ValidationResult(
                    is_match=ai_result['is_match'],
                    confidence=ai_result.get('confidence', 0.9),
                    reasoning=ai_result.get('reasoning', 'AI decision'),
                    cache_hit=False,
                    response_time_ms=100  # Approximate
                )
                results.append((idx, result))
                
                # Cache the result
                if self.cache_manager:
                    cache_key = self._generate_cache_key(vertical, business_types, business_title)
                    self.cache_manager.set_validation(cache_key, ai_result['is_match'])
                    
            except Exception as e:
                # Individual item failed, use fallback
                is_match = self._fallback_validation(vertical, business_types, business_title)
                result = ValidationResult(
                    is_match=is_match,
                    confidence=0.5,
                    reasoning=f"Individual fallback: {str(e)}",
                    cache_hit=False,
                    response_time_ms=10
                )
                results.append((idx, result))
        
        return results
    
    def _create_validation_prompt(self, vertical: str, business_types: List[str], 
                                business_title: str = "") -> str:
        """Create a validation prompt for the AI."""
        types_str = ", ".join(f'"{t}"' for t in business_types)
        
        prompt = f"""Search query: "{vertical}"
Business types: [{types_str}]"""
        
        if business_title:
            prompt += f'\nBusiness title: "{business_title}"'
        
        prompt += "\n\nDoes this business match the search query? Answer YES or NO."
        
        return prompt
    
    def _fallback_validation(self, vertical: str, business_types: List[str], 
                           business_title: str = "") -> bool:
        """
        Rule-based fallback validation when AI is not available.
        This replicates the original logic from serper.py but checks 'types' instead of 'categories'.
        """
        vertical_lower = vertical.lower()
        
        # Check business types (this is the main improvement - checking the right field)
        for business_type in business_types:
            type_lower = business_type.lower()
            
            # Direct match
            if vertical_lower in type_lower:
                return True
            
            # Word-by-word match
            if any(word in type_lower for word in vertical_lower.split()):
                return True
        
        # Check business title as secondary validation
        if business_title:
            title_lower = business_title.lower()
            if vertical_lower in title_lower:
                return True
            if any(word in title_lower for word in vertical_lower.split()):
                return True
        
        return False
    
    def _generate_cache_key(self, vertical: str, business_types: List[str], 
                          business_title: str = "") -> str:
        """Generate a cache key for the validation request."""
        # Create a consistent key by sorting business types
        key_data = f"{vertical.lower()}|{sorted([t.lower() for t in business_types])}|{business_title.lower()}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_stats(self) -> Dict:
        """Get validation statistics."""
        stats = {
            'enabled': self.enabled,
            'groq_available': GROQ_AVAILABLE,
            'has_api_key': bool(settings.GROQ_API_KEY),
            'cache_enabled': self.cache_manager is not None,
            'batch_validation_enabled': self.batch_validator is not None
        }
        
        if self.cache_manager:
            cache_stats = self.cache_manager.get_cache_stats()
            stats.update(cache_stats)
        
        if self.batch_validator:
            batch_stats = self.batch_validator.get_stats()
            stats['batch_size'] = batch_stats.get('batch_size', 0)
        
        return stats