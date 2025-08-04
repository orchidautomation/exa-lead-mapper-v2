"""
Data models for the Business Mapper API v2.0
"""
from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class SearchRequest(BaseModel):
    """Request model for the main search endpoint."""
    verticals: List[str] = Field(..., min_items=1, description="List of business types to search")
    locations: List[str] = Field(..., min_items=1, description="List of locations to search")
    zoom: int = Field(14, ge=1, le=21, description="Map zoom level")
    max_pages: Optional[int] = Field(None, ge=1, le=8, description="Maximum number of result pages")
    output_format: str = Field('json', pattern='^(json|csv)$', description="Output format")
    
    # Auto-expansion parameters
    auto_expand: bool = Field(False, description="Automatically expand broad verticals")
    expansion_mode: str = Field('balanced', pattern='^(minimal|balanced|comprehensive)$')
    expansion_priority: str = Field('balanced', pattern='^(cost_control|balanced|coverage)$')
    max_credit_budget: Optional[int] = Field(None, ge=1, description="Maximum credits to spend")
    max_cost_multiplier: Optional[float] = Field(None, ge=1.0, le=10.0)
    
    # Context for AI expansion
    search_intent: Optional[str] = Field(None, max_length=500, 
        description="Context for AI expansion (e.g., 'facilities with real grass for lawn maintenance')")
    
    # Reviews parameters
    include_reviews: bool = Field(False, description="Include raw reviews data")
    analyze_reviews: bool = Field(False, description="Enable AI-powered review analysis")
    reviews_sort_by: str = Field('mostRelevant', pattern='^(highestRating|mostRelevant|newest|lowestRating)$')
    max_reviews_analyze: int = Field(50, ge=20, le=100, description="Max reviews to analyze per business")
    min_reviews_for_analysis: int = Field(20, ge=5, le=50, description="Min reviews for AI analysis")
    
    @validator('verticals', 'locations')
    def validate_non_empty_list(cls, v):
        cleaned = [item.strip() for item in v if item.strip()]
        if not cleaned:
            raise ValueError("At least one valid item is required")
        return cleaned
    
    @validator('analyze_reviews')
    def validate_analyze_reviews(cls, v, values):
        if v and not values.get('include_reviews', False):
            raise ValueError("analyze_reviews=true requires include_reviews=true")
        return v


class PlaceResult(BaseModel):
    """Model for a single place/business result."""
    name: str
    placeId: str
    cid: Optional[str] = None
    rating: float = Field(0.0, ge=0.0, le=5.0)
    reviews: int = Field(0, ge=0)
    type: str
    address: str
    street_address: Optional[str] = None
    city: str
    stateCode: str
    zip_code: Optional[str] = None
    latitude: float
    longitude: float
    population_type: Optional[str] = Field(None, description="urban, suburban, rural, dense_urban")
    price_level: Optional[str] = None
    website: Optional[str] = None
    phoneNumber: Optional[str] = None
    openingHours: Optional[Dict] = None
    thumbnailUrl: Optional[str] = None
    vertical: str
    mapsUrl: str
    reviews_data: Optional[List[Dict]] = None
    reviews_analysis: Optional[Union['ReviewsAnalysis', 'ReviewsAnalysisFallback']] = None


class ReviewInsight(BaseModel):
    """Model for individual review insight."""
    insight: str = Field(..., min_length=10, max_length=200)
    confidence: float = Field(..., ge=0.7, le=1.0)
    frequency: int = Field(..., ge=1)
    supporting_quotes: List[str] = Field(..., min_items=1, max_items=3)


class ReviewsAnalysis(BaseModel):
    """Model for comprehensive reviews analysis results - business agnostic."""
    analysis_status: str = Field(default="completed")
    total_reviews_analyzed: int = Field(..., ge=5)
    average_rating: float = Field(..., ge=1.0, le=5.0)
    rating_distribution: Dict[str, int]
    
    # Sentiment analysis
    overall_sentiment: str = Field(..., pattern='^(positive|negative|mixed)$')
    sentiment_score: float = Field(..., ge=-100.0, le=100.0, description="-100=very negative, 0=neutral, 100=very positive")
    
    # Key insights
    top_positives: List[ReviewInsight] = Field(..., min_items=0, max_items=10, description="Most praised aspects")
    top_negatives: List[ReviewInsight] = Field(..., min_items=0, max_items=10, description="Most criticized aspects")
    
    # Business intelligence
    key_themes: List[str] = Field(..., min_items=0, max_items=10, description="Main themes from reviews")
    business_opportunities: List[str] = Field(..., min_items=0, max_items=5, description="Opportunities based on customer feedback")
    summary: str = Field(..., min_length=50, max_length=500, description="AI-generated summary of reviews")
    
    # Metadata
    analysis_confidence: float = Field(..., ge=0.0, le=1.0)
    analysis_timestamp: str
    reviews_date_range: Dict[str, str]


class ReviewsAnalysisFallback(BaseModel):
    """Fallback when insufficient reviews for AI analysis."""
    analysis_status: str = Field(default="insufficient_reviews")
    total_reviews_available: int = Field(..., ge=0)
    minimum_required: int = Field(default=20)
    message: str = Field(default="Insufficient reviews for AI analysis")
    analysis_timestamp: str


class TermContribution(BaseModel):
    """Model for individual term contribution data."""
    term: str
    total_results: int
    unique_results: int
    contribution_rate: float
    cost_efficiency: float
    discovery_value: float
    overlaps_with: List[str]


class ContributionAnalysis(BaseModel):
    """Model for term contribution analysis."""
    session_id: str
    overall_duplicate_rate: float
    total_unique_results: int
    term_contributions: List[TermContribution]
    recommendations: List[str]
    analysis_time_ms: int


class SearchResponse(BaseModel):
    """Response model for search results."""
    results: List[PlaceResult]
    total_credits_used: int
    cost_of_credits: float
    unique_businesses_found: int
    raw_results_count: int
    search_metadata: Dict = Field(default_factory=dict)
    term_contribution_analysis: Optional[ContributionAnalysis] = None
    reviews_metadata: Optional[Dict] = None


class APIKeyRequest(BaseModel):
    """Request model for API key generation."""
    description: Optional[str] = Field(None, description="Description for the API key")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """Response model for API key generation."""
    api_key: str
    created_at: datetime
    description: Optional[str] = None
    expires_at: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = "healthy"
    version: str = "2.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str] = Field(default_factory=dict)