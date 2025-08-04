import json
import logging
import time
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

from models.schemas import ReviewsAnalysis, ReviewInsight, ReviewsAnalysisFallback
from config.settings import settings

logger = logging.getLogger(__name__)

class ReviewsAnalyzer:
    """
    AI-powered reviews analyzer using Groq for extracting structured insights.
    
    Extracts top praises and problems from customer reviews with:
    - Robust prompt engineering for quality insights
    - Structured output validation with Pydantic
    - Multilingual support
    - Aggressive caching to minimize costs
    """
    
    # AI Configuration
    MODEL_NAME = "moonshotai/kimi-k2-instruct"  # Supports structured outputs!
    FALLBACK_MODEL = "llama-3.3-70b-versatile"  # Fallback with JSON Object Mode
    MAX_TOKENS = 2000
    TEMPERATURE = 0.1  # Low temperature for consistent structured output
    
    # Quality thresholds
    DEFAULT_MIN_REVIEWS_FOR_ANALYSIS = 20  # Default threshold for reliable analysis
    MIN_INSIGHT_CONFIDENCE = 0.7
    MAX_RETRIES = 2
    
    # Caching
    CACHE_TTL_HOURS = 168  # 7 days for AI analysis
    
    def __init__(self, groq_client=None, cache_manager=None):
        """
        Initialize the reviews analyzer.
        
        Args:
            groq_client: Optional Groq client instance
            cache_manager: Cache manager for storing analysis results
        """
        self.cache_manager = cache_manager
        self.groq_client = groq_client
        self.enabled = False
        
        # Initialize Groq client if not provided
        if not self.groq_client and GROQ_AVAILABLE and settings.GROQ_API_KEY:
            try:
                # Initialize with just api_key - no proxy settings
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                self.enabled = True
                logger.info("🤖 Reviews analyzer enabled with Groq")
            except TypeError as e:
                # Try without any extra arguments if the client doesn't accept them
                if "proxies" in str(e):
                    try:
                        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                        self.enabled = True
                        logger.info("🤖 Reviews analyzer enabled with Groq (fallback)")
                    except Exception as e2:
                        logger.warning(f"Failed to initialize Groq client (fallback): {str(e2)}")
                else:
                    logger.warning(f"Failed to initialize Groq client: {str(e)}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {str(e)}")
        elif self.groq_client:
            self.enabled = True
        else:
            logger.warning("Reviews analyzer disabled - no Groq client available")
    
    def analyze_reviews(self, reviews: List[Dict], place_name: str, min_reviews: int = None) -> Optional[ReviewsAnalysis]:
        """
        Analyze reviews to extract top praises and problems.
        
        Args:
            reviews: List of review dictionaries from Serper
            place_name: Name of the business for context
            min_reviews: Minimum reviews required (defaults to DEFAULT_MIN_REVIEWS_FOR_ANALYSIS)
            
        Returns:
            ReviewsAnalysis object, ReviewsAnalysisFallback, or None if analysis fails
        """
        if not self.enabled:
            logger.warning("Reviews analyzer not enabled")
            return None
            
        # Use provided minimum or default
        min_required = min_reviews if min_reviews is not None else self.DEFAULT_MIN_REVIEWS_FOR_ANALYSIS
        
        if len(reviews) < min_required:
            logger.info(f"Insufficient reviews for analysis: {len(reviews)} < {min_required}")
            return self._create_fallback_response(len(reviews), min_required)
        
        try:
            # Check cache first
            content_hash = self._generate_content_hash(reviews)
            cache_key = f"analysis_{content_hash}"
            
            if self.cache_manager:
                cached_analysis = self._get_cached_analysis(cache_key)
                if cached_analysis:
                    logger.debug(f"Using cached analysis for {place_name}")
                    return cached_analysis
            
            # Prepare reviews text for analysis
            reviews_text = self._prepare_reviews_text(reviews)
            
            # Generate AI analysis
            analysis_result = self._generate_analysis(reviews_text, place_name)
            
            if analysis_result:
                # Cache the result
                if self.cache_manager:
                    self._cache_analysis(cache_key, analysis_result)
                
                logger.info(f"Generated reviews analysis for {place_name}: "
                           f"{len(analysis_result.top_positives)} positives, "
                           f"{len(analysis_result.top_negatives)} negatives")
                return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing reviews for {place_name}: {str(e)}")
        
        return None
    
    def _generate_analysis(self, reviews_text: str, place_name: str) -> Optional[ReviewsAnalysis]:
        """
        Generate AI analysis using Groq with robust prompting.
        
        Args:
            reviews_text: Preprocessed reviews text
            place_name: Business name for context
            
        Returns:
            ReviewsAnalysis object or None
        """
        if not self.groq_client:
            return None
        
        prompt = self._build_analysis_prompt(reviews_text, place_name)
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                # Try structured output first with supported models
                if self.MODEL_NAME in ["moonshotai/kimi-k2-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct"]:
                    response = self._create_structured_response(prompt, reviews_text, place_name)
                else:
                    # Fallback to JSON Object Mode for other models
                    response = self._create_json_mode_response(prompt)
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse and validate the JSON response
                analysis_data = self._parse_and_validate_response(response_text)
                
                if analysis_data:
                    return analysis_data
                else:
                    logger.warning(f"Invalid analysis response on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(1)  # Brief delay before retry
        
        logger.error(f"Failed to generate analysis after {self.MAX_RETRIES + 1} attempts")
        return None
    
    def _create_structured_response(self, prompt: str, reviews_text: str, place_name: str):
        """Create response using Structured Outputs (for supported models)."""
        # Groq SDK uses strict=True format for JSON schema
        return self.groq_client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a business analyst extracting insights from customer reviews for {place_name}. Analyze the reviews and provide structured insights."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reviews_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "total_reviews_analyzed": {"type": "integer"},
                            "average_rating": {"type": "number"},
                            "rating_distribution": {
                                "type": "object",
                                "properties": {
                                    "1": {"type": "integer"},
                                    "2": {"type": "integer"},
                                    "3": {"type": "integer"},
                                    "4": {"type": "integer"},
                                    "5": {"type": "integer"}
                                },
                                "required": ["1", "2", "3", "4", "5"],
                                "additionalProperties": False
                            },
                            "overall_sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "mixed"]
                            },
                            "sentiment_score": {"type": "number"},
                            "top_positives": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "insight": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "frequency": {"type": "integer"},
                                        "supporting_quotes": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["insight", "confidence", "frequency", "supporting_quotes"],
                                    "additionalProperties": False
                                }
                            },
                            "top_negatives": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "insight": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "frequency": {"type": "integer"},
                                        "supporting_quotes": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["insight", "confidence", "frequency", "supporting_quotes"],
                                    "additionalProperties": False
                                }
                            },
                            "key_themes": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "business_opportunities": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "summary": {"type": "string"},
                            "analysis_confidence": {"type": "number"},
                            "analysis_timestamp": {"type": "string"},
                            "reviews_date_range": {
                                "type": "object",
                                "properties": {
                                    "oldest": {"type": "string"},
                                    "newest": {"type": "string"}
                                },
                                "required": ["oldest", "newest"],
                                "additionalProperties": False
                            }
                        },
                        "required": [
                            "total_reviews_analyzed", "average_rating", "rating_distribution",
                            "overall_sentiment", "sentiment_score", "top_positives",
                            "top_negatives", "key_themes", "business_opportunities",
                            "summary", "analysis_confidence", "analysis_timestamp",
                            "reviews_date_range"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE
        )
    
    def _create_json_mode_response(self, prompt: str):
        """Create response using JSON Object Mode (fallback for unsupported models)."""
        return self.groq_client.chat.completions.create(
            model=self.FALLBACK_MODEL if self.MODEL_NAME == "llama3-8b-8192" else self.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE,
            top_p=0.9
        )
    
    def _build_analysis_prompt(self, reviews_text: str, place_name: str) -> str:
        """
        Build the robust AI prompt for reviews analysis.
        
        Args:
            reviews_text: Preprocessed reviews text
            place_name: Business name
            
        Returns:
            Complete analysis prompt
        """
        # For structured outputs, we can use a cleaner prompt
        if self.MODEL_NAME in ["moonshotai/kimi-k2-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct"]:
            return f'''Analyze these customer reviews for "{place_name}" and extract actionable business insights.

Focus on:
- Specific, actionable insights (not generic statements)
- Patterns that appear in multiple reviews (frequency 2+)
- Business-impacting themes
- Brief supporting quotes (max 3 per insight)
- Confidence scores above 0.7

REVIEWS TO ANALYZE:
{reviews_text}'''
        
        # Original detailed prompt for JSON Object Mode
        return f'''TASK: Analyze customer reviews for "{place_name}" and extract comprehensive business insights.

CRITICAL REQUIREMENTS:
1. Each insight must be SPECIFIC and ACTIONABLE - no generic statements
2. Minimum confidence threshold: 0.7 (if below, don't include)  
3. Must appear in multiple reviews to qualify (frequency ≥ 2)
4. Focus on business-impacting themes, not trivial mentions
5. Extract 1-3 brief supporting quotes as evidence (MAXIMUM 3 QUOTES PER INSIGHT)
6. Maximum 10 positive and 10 negative insights each
7. All confidence scores must be ≥ 0.7 (if lower, exclude the insight)
8. Determine overall sentiment and score objectively

QUALITY STANDARDS:
❌ BAD EXAMPLES (too generic/weak):
- "Good service" → Too vague
- "Nice place" → Not specific
- "Not bad" → Not impactful

✅ GOOD EXAMPLES (specific/actionable):
- "Staff remembers regular customers' names and preferences"
- "Parking extremely limited during peak hours, causing 10-15 min delays"
- "Pricing 20-30% higher than similar businesses in the area"
- "Equipment frequently out of service, affecting customer experience"

ANALYSIS PROCESS:
1. Read ALL reviews thoroughly
2. Identify RECURRING themes (positive & negative)
3. Extract SPECIFIC, ACTIONABLE insights only
4. Calculate sentiment score based on overall tone (0=very negative, 100=very positive)
5. Identify key business themes and opportunities
6. Create a concise summary of the overall customer experience

RESPONSE FORMAT: Return ONLY raw JSON (no markdown, no code blocks, no ```json``` wrapping) matching this exact structure:
{{
  "total_reviews_analyzed": <number>,
  "average_rating": <float>,
  "rating_distribution": {{"1": <count>, "2": <count>, "3": <count>, "4": <count>, "5": <count>}},
  "overall_sentiment": "<positive|negative|mixed>",
  "sentiment_score": <float 0-100>,
  "top_positives": [
    {{
      "insight": "<specific positive insight>",
      "confidence": <float 0.7-1.0>,
      "frequency": <int ≥2>,
      "supporting_quotes": ["<quote1>", "<quote2>"]
    }}
  ],
  "top_negatives": [
    {{
      "insight": "<specific negative insight>", 
      "confidence": <float 0.7-1.0>,
      "frequency": <int ≥2>,
      "supporting_quotes": ["<quote1>", "<quote2>"]
    }}
  ],
  "key_themes": ["<theme1>", "<theme2>", "<theme3>"],
  "business_opportunities": ["<opportunity1>", "<opportunity2>"],
  "summary": "<concise 50-500 character summary of overall customer sentiment and experience>",
  "analysis_confidence": <overall_confidence_float>,
  "analysis_timestamp": "{datetime.now().isoformat()}Z",
  "reviews_date_range": {{"oldest": "<YYYY-MM-DD>", "newest": "<YYYY-MM-DD>"}}
}}

REVIEWS TO ANALYZE:
{reviews_text}

Take a deep breath and work through this systematically. Focus on business-critical insights that would help improve operations. This analysis is crucial for business improvement.'''
    
    def _prepare_reviews_text(self, reviews: List[Dict]) -> str:
        """
        Prepare reviews text for AI analysis.
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            Formatted reviews text
        """
        formatted_reviews = []
        
        for i, review in enumerate(reviews, 1):
            # Get review content (prefer translated if available)
            snippet = review.get('translatedSnippet') or review.get('snippet', '')
            rating = review.get('rating', 0)
            date = review.get('date', 'unknown')
            
            # Clean and format
            snippet = snippet.strip()
            if snippet:
                formatted_reviews.append(
                    f"Review #{i} ({rating}/5 stars, {date}):\n{snippet}\n"
                )
        
        return "\n".join(formatted_reviews)
    
    def _parse_and_validate_response(self, response_text: str) -> Optional[ReviewsAnalysis]:
        """
        Parse and validate AI response into ReviewsAnalysis object.
        
        Args:
            response_text: Raw AI response
            
        Returns:
            Validated ReviewsAnalysis object or None
        """
        try:
            # Handle multiple JSON extraction strategies
            json_text = None
            
            # Strategy 1: Look for JSON wrapped in markdown code blocks
            if '```json' in response_text:
                start_marker = response_text.find('```json') + 7
                end_marker = response_text.find('```', start_marker)
                if end_marker > start_marker:
                    json_text = response_text[start_marker:end_marker].strip()
            
            # Strategy 2: Look for JSON wrapped in regular code blocks
            elif '```' in response_text and not json_text:
                start_marker = response_text.find('```') + 3
                end_marker = response_text.find('```', start_marker)
                if end_marker > start_marker:
                    potential_json = response_text[start_marker:end_marker].strip()
                    if potential_json.startswith('{'):
                        json_text = potential_json
            
            # Strategy 3: Extract raw JSON (original method)
            if not json_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > 0:
                    json_text = response_text[json_start:json_end]
            
            if not json_text:
                logger.warning("No valid JSON found in AI response")
                return None
            
            analysis_data = json.loads(json_text)
            
            # Validate with Pydantic
            analysis = ReviewsAnalysis(**analysis_data)
            
            # Additional quality validation
            analysis = self._validate_insights_quality(analysis)
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in AI response: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Validation failed: {str(e)}")
            return None
    
    def _validate_insights_quality(self, analysis: ReviewsAnalysis) -> ReviewsAnalysis:
        """
        Additional quality validation for insights.
        
        Args:
            analysis: ReviewsAnalysis object
            
        Returns:
            Filtered ReviewsAnalysis with quality insights only
        """
        def filter_insights(insights: List[ReviewInsight]) -> List[ReviewInsight]:
            filtered = []
            for insight in insights:
                # Check minimum confidence and frequency
                if insight.confidence >= self.MIN_INSIGHT_CONFIDENCE and insight.frequency >= 2:
                    # Check insight quality (length, specificity)
                    if len(insight.insight) >= 15 and not self._is_generic_insight(insight.insight):
                        filtered.append(insight)
            return filtered
        
        # Filter positives and negatives
        filtered_positives = filter_insights(analysis.top_positives)
        filtered_negatives = filter_insights(analysis.top_negatives)
        
        # Update the analysis
        analysis.top_positives = filtered_positives
        analysis.top_negatives = filtered_negatives
        
        # Recalculate confidence based on quality
        total_insights = len(filtered_positives) + len(filtered_negatives)
        if total_insights > 0:
            avg_confidence = sum(
                insight.confidence for insight in filtered_positives + filtered_negatives
            ) / total_insights
            analysis.analysis_confidence = min(analysis.analysis_confidence, avg_confidence)
        else:
            analysis.analysis_confidence = 0.0
        
        return analysis
    
    def _is_generic_insight(self, insight: str) -> bool:
        """
        Check if an insight is too generic to be useful.
        
        Args:
            insight: Insight text
            
        Returns:
            True if insight is generic
        """
        generic_phrases = [
            'good service', 'great food', 'nice place', 'friendly staff',
            'bad service', 'poor quality', 'terrible experience', 'not good',
            'excellent', 'amazing', 'awesome', 'horrible', 'awful'
        ]
        
        insight_lower = insight.lower()
        return any(phrase in insight_lower for phrase in generic_phrases)
    
    def _generate_content_hash(self, reviews: List[Dict]) -> str:
        """
        Generate hash of reviews content for caching.
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            Content hash string
        """
        # Create a stable representation of the reviews content
        content_items = []
        for review in reviews:
            snippet = review.get('translatedSnippet') or review.get('snippet', '')
            rating = review.get('rating', 0)
            content_items.append(f"{rating}:{snippet}")
        
        content_string = "|".join(sorted(content_items))
        return hashlib.md5(content_string.encode()).hexdigest()
    
    def _get_cached_analysis(self, cache_key: str) -> Optional[ReviewsAnalysis]:
        """
        Retrieve cached analysis.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached ReviewsAnalysis or None
        """
        try:
            if hasattr(self.cache_manager, 'get_analysis_cache'):
                cached_data = self.cache_manager.get_analysis_cache(cache_key)
                if cached_data:
                    return ReviewsAnalysis(**cached_data)
            else:
                # Fallback to generic cache
                cached_data = getattr(self.cache_manager, 'get', lambda x: None)(f"analysis_{cache_key}")
                if cached_data:
                    return ReviewsAnalysis(**cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
        
        return None
    
    def _cache_analysis(self, cache_key: str, analysis: ReviewsAnalysis) -> None:
        """
        Cache analysis results.
        
        Args:
            cache_key: Cache key
            analysis: ReviewsAnalysis object to cache
        """
        try:
            analysis_data = analysis.dict()
            
            if hasattr(self.cache_manager, 'set_analysis_cache'):
                self.cache_manager.set_analysis_cache(
                    cache_key, analysis_data, ttl_hours=self.CACHE_TTL_HOURS
                )
            else:
                # Fallback to generic cache
                if hasattr(self.cache_manager, 'set'):
                    self.cache_manager.set(
                        f"analysis_{cache_key}", analysis_data,
                        ttl=self.CACHE_TTL_HOURS * 3600
                    )
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")
    
    def _create_fallback_response(self, available_reviews: int, min_required: int = None) -> ReviewsAnalysisFallback:
        """
        Create a safe fallback response when insufficient reviews for AI analysis.
        
        Args:
            available_reviews: Number of reviews available
            min_required: Minimum reviews required (defaults to DEFAULT_MIN_REVIEWS_FOR_ANALYSIS)
            
        Returns:
            ReviewsAnalysisFallback with helpful information
        """
        min_req = min_required if min_required is not None else self.DEFAULT_MIN_REVIEWS_FOR_ANALYSIS
        
        return ReviewsAnalysisFallback(
            total_reviews_available=available_reviews,
            minimum_required=min_req,
            message=f"Only {available_reviews} reviews available. AI analysis requires at least {min_req} reviews for reliable insights.",
            analysis_timestamp=datetime.now().isoformat()
        )