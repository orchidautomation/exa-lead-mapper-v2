"""
Cost-aware AI-powered vertical expansion service.

This service intelligently expands broad search terms (like "roofing") into comprehensive
vertical lists while respecting credit budgets and location density constraints.
"""

import hashlib
import json
import logging
import time
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

from config.settings import settings
from services.term_contribution_analyzer import TermContributionAnalyzer

logger = logging.getLogger(__name__)

class ExpansionPriority(Enum):
    """Expansion priority modes balancing cost vs coverage."""
    COST_CONTROL = "cost_control"      # Minimize cost, basic coverage
    BALANCED = "balanced"              # Optimize cost/coverage ratio
    COVERAGE = "coverage"              # Maximize coverage within reason

class ExpansionMode(Enum):
    """Expansion depth modes."""
    MINIMAL = "minimal"                # 2-3 core terms
    BALANCED = "balanced"              # 4-5 common terms
    COMPREHENSIVE = "comprehensive"    # 6-8 specialized terms

@dataclass
class VerticalSuggestion:
    """A suggested vertical with metadata."""
    term: str
    relevance_score: float         # 0.0 to 1.0
    unique_coverage_estimate: float # Estimated unique businesses this term finds
    cost_efficiency_score: float   # Results per credit ratio
    reasoning: str

@dataclass
class ExpansionResult:
    """Complete expansion analysis and recommendations."""
    original_vertical: str
    suggested_verticals: List[VerticalSuggestion]
    selected_verticals: List[str]
    expansion_reasoning: str
    cost_analysis: Dict
    coverage_analysis: Dict
    location_context: Optional[str]
    cache_hit: bool
    response_time_ms: int
    ai_enabled: bool

class CostAwareExpander:
    """
    Intelligently expands broad search terms into comprehensive vertical lists
    while respecting credit budgets and location density patterns.
    """
    
    # Industry knowledge base for common expansions
    INDUSTRY_EXPANSIONS = {
        'roofing': [
            'roofing companies', 'roofers', 'roofing contractors', 'roof repair',
            'roofing services', 'roof installation', 'commercial roofing', 'residential roofing'
        ],
        'hvac': [
            'hvac contractors', 'air conditioning repair', 'heating and cooling',
            'hvac companies', 'ac repair', 'furnace repair', 'hvac services', 'hvac installation'
        ],
        'legal': [
            'lawyers', 'attorneys', 'law firms', 'legal services', 'personal injury lawyers',
            'family lawyers', 'criminal defense lawyers', 'immigration lawyers'
        ],
        'plumbing': [
            'plumbers', 'plumbing companies', 'plumbing contractors', 'plumbing services',
            'emergency plumbers', 'drain cleaning', 'pipe repair', 'water heater repair'
        ],
        'automotive': [
            'auto repair', 'car repair', 'automotive services', 'auto mechanics',
            'brake repair', 'oil change', 'tire shops', 'auto body shops'
        ],
        'dental': [
            'dentists', 'dental offices', 'family dentistry', 'cosmetic dentistry',
            'orthodontists', 'dental clinics', 'oral surgeons', 'pediatric dentists'
        ],
        'restaurant': [
            'restaurants', 'dining', 'food', 'cafes', 'fast food', 'fine dining',
            'pizza', 'italian restaurants', 'mexican restaurants', 'chinese restaurants'
        ],
        'golf': [
            'golf courses', 'golf clubs', 'country clubs', 'public golf courses',
            'private golf courses', 'golf resorts', 'municipal golf courses', 'championship golf courses'
        ]
    }
    
    def __init__(self, geo_optimizer=None, cache_manager=None):
        """
        Initialize cost-aware expander.
        
        Args:
            geo_optimizer: GeographicOptimizer for location intelligence
            cache_manager: Optional cache manager for storing expansion results
        """
        self.geo_optimizer = geo_optimizer
        self.cache_manager = cache_manager
        self.groq_client = None
        self.enabled = False
        
        # Initialize contribution analyzer if database available
        # Temporarily disabled - missing database methods
        # if cache_manager:
        #     self.contribution_analyzer = TermContributionAnalyzer(cache_manager)
        # else:
        #     self.contribution_analyzer = None
        self.contribution_analyzer = None
        
        # Initialize Groq client if available
        if GROQ_AVAILABLE and settings.GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                self.enabled = True
                logger.info("🚀 Cost-aware vertical expansion enabled with Groq")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq for expansion: {str(e)}")
        else:
            if not GROQ_AVAILABLE:
                logger.warning("Groq library not available for vertical expansion")
            elif not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not configured. Vertical expansion disabled.")
    
    def expand_vertical(self, 
                       vertical: str,
                       location: Optional[str] = None,
                       expansion_mode: str = "balanced",
                       expansion_priority: str = "balanced",
                       max_credit_budget: Optional[int] = None,
                       max_cost_multiplier: Optional[float] = None,
                       search_intent: Optional[str] = None) -> ExpansionResult:
        """
        Expand a broad vertical into comprehensive search terms with cost awareness.
        
        Args:
            vertical: Broad search term (e.g., "roofing", "legal")
            location: Location context for cost calculations
            expansion_mode: Depth of expansion (minimal, balanced, comprehensive)
            expansion_priority: Priority (cost_control, balanced, coverage)
            max_credit_budget: Maximum credits to spend on expanded search
            max_cost_multiplier: Maximum cost increase vs original (e.g., 3.0 = 3x)
            search_intent: Optional context/intent for AI expansion (e.g., 'facilities with real grass for lawn maintenance')
            
        Returns:
            ExpansionResult with selected terms and cost analysis
        """
        start_time = time.time()
        
        # Normalize inputs
        vertical_clean = vertical.strip().lower()
        
        # Check if expansion is even needed
        if not self._needs_expansion(vertical_clean):
            return self._create_no_expansion_result(vertical, start_time)
        
        # Get location intelligence for cost calculations
        location_intel = self._get_location_intelligence(location)
        
        # Check cache first
        cache_key = self._generate_cache_key(vertical_clean, location, expansion_mode, expansion_priority)
        if self.cache_manager:
            cached_result = self._get_cached_expansion(cache_key)
            if cached_result:
                response_time = int((time.time() - start_time) * 1000)
                cached_result.response_time_ms = response_time
                cached_result.cache_hit = True
                return cached_result
        
        # Get expansion suggestions
        if self.enabled:
            suggestions = self._get_ai_suggestions(vertical_clean, location, expansion_mode, search_intent)
        else:
            suggestions = self._get_fallback_suggestions(vertical_clean, expansion_mode)
        
        # Apply cost constraints and select optimal terms
        selected_terms = self._apply_cost_constraints(
            suggestions, location_intel, expansion_priority, 
            max_credit_budget, max_cost_multiplier
        )
        
        # Generate cost and coverage analysis
        cost_analysis = self._analyze_costs(selected_terms, location_intel, vertical)
        coverage_analysis = self._analyze_coverage(vertical, selected_terms, suggestions)
        
        response_time = int((time.time() - start_time) * 1000)
        
        result = ExpansionResult(
            original_vertical=vertical,
            suggested_verticals=suggestions,
            selected_verticals=selected_terms,
            expansion_reasoning=self._generate_expansion_reasoning(
                vertical, selected_terms, expansion_priority, cost_analysis
            ),
            cost_analysis=cost_analysis,
            coverage_analysis=coverage_analysis,
            location_context=location,
            cache_hit=False,
            response_time_ms=response_time,
            ai_enabled=self.enabled
        )
        
        # Cache the result
        if self.cache_manager:
            self._cache_expansion(cache_key, result)
        
        return result
    
    def estimate_expansion_cost(self, 
                               vertical: str,
                               location: Optional[str] = None,
                               expansion_modes: List[str] = None) -> Dict:
        """
        Estimate costs for different expansion scenarios without actually expanding.
        
        Args:
            vertical: Vertical to analyze
            location: Location for cost calculations
            expansion_modes: List of modes to analyze (default: all)
            
        Returns:
            Dictionary with cost estimates for each mode
        """
        if expansion_modes is None:
            expansion_modes = ["minimal", "balanced", "comprehensive"]
        
        location_intel = self._get_location_intelligence(location)
        base_cost = self._calculate_base_cost(location_intel)
        
        estimates = {
            "original": {
                "terms": 1,
                "estimated_credits": base_cost,
                "estimated_cost_usd": base_cost * 0.001,
                "estimated_results": self._estimate_results_for_terms([vertical], location_intel)
            }
        }
        
        for mode in expansion_modes:
            # Get quick suggestions without full AI analysis
            if vertical.lower() in self.INDUSTRY_EXPANSIONS:
                suggestions = self._get_fallback_suggestions(vertical.lower(), mode)
            else:
                # For unknown verticals, estimate based on typical patterns
                term_counts = {"minimal": 3, "balanced": 5, "comprehensive": 7}
                suggestions = [
                    VerticalSuggestion(f"{vertical} option {i}", 0.8, 0.2, 0.7, "estimated")
                    for i in range(term_counts.get(mode, 5))
                ]
            
            selected_terms = [s.term for s in suggestions]
            cost = len(selected_terms) * base_cost
            
            estimates[mode] = {
                "terms": len(selected_terms),
                "estimated_credits": cost,
                "estimated_cost_usd": cost * 0.001,
                "estimated_results": self._estimate_results_for_terms(selected_terms, location_intel),
                "cost_multiplier": cost / base_cost if base_cost > 0 else 1
            }
        
        return estimates
    
    def _needs_expansion(self, vertical: str) -> bool:
        """Check if a vertical term needs expansion."""
        # Don't expand if already specific
        specific_indicators = [
            'companies', 'contractors', 'services', 'repair', 'installation',
            'shops', 'stores', 'clinics', 'offices'
        ]
        
        if any(indicator in vertical for indicator in specific_indicators):
            return False
        
        # Check if it's a known broad category
        return vertical in self.INDUSTRY_EXPANSIONS or len(vertical.split()) <= 2
    
    def _get_location_intelligence(self, location: Optional[str]) -> Dict:
        """Get location intelligence for cost calculations."""
        if not location or not self.geo_optimizer:
            # Default intelligence for unknown locations
            return {
                'location_type': 'unknown',
                'max_pages': 3,
                'cost_per_term': 9,  # 3 pages * 3 credits
                'business_density': 'medium'
            }
        
        # Get location type from geographic optimizer
        location_type = self.geo_optimizer.classify_location_type(location)
        pagination_rules = self.geo_optimizer.get_pagination_rules(location_type)
        
        return {
            'location_type': location_type.value,
            'max_pages': pagination_rules['max_pages'],
            'cost_per_term': pagination_rules['max_pages'] * 3,  # 3 credits per page
            'business_density': location_type.value
        }
    
    def _get_ai_suggestions(self, vertical: str, location: Optional[str], mode: str, search_intent: Optional[str] = None) -> List[VerticalSuggestion]:
        """Get AI-powered expansion suggestions."""
        try:
            prompt = self._create_expansion_prompt(vertical, location, mode, search_intent)
            
            response = self.groq_client.chat.completions.create(
                model=settings.AI_VALIDATION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a business search expert. Suggest search terms that businesses "
                            "actually use to list themselves. Focus on real-world terminology and "
                            "industry-specific language that would appear in business directories."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            ai_response = response.choices[0].message.content.strip()
            logger.debug(f"AI expansion response: {ai_response}")
            return self._parse_ai_suggestions(ai_response, vertical)
            
        except Exception as e:
            logger.error(f"AI expansion failed: {str(e)}")
            return self._get_fallback_suggestions(vertical, mode)
    
    def _create_expansion_prompt(self, vertical: str, location: Optional[str], mode: str, search_intent: Optional[str] = None) -> str:
        """Create AI prompt for vertical expansion."""
        mode_descriptions = {
            "minimal": "2-3 essential terms",
            "balanced": "4-5 common terms", 
            "comprehensive": "6-8 specialized terms"
        }
        
        location_context = f" in {location}" if location else ""
        mode_desc = mode_descriptions.get(mode, "4-5 terms")
        
        prompt = f"""Suggest {mode_desc} for finding "{vertical}" businesses{location_context}.

Consider how businesses actually list themselves in directories:
- Different terminology variations
- Industry-specific terms
- Service-specific terms
- Common business naming patterns

Vertical: "{vertical}"
{f'Location: {location}' if location else ''}
Mode: {mode}
{f'Search Intent: {search_intent}' if search_intent else ''}

{f'IMPORTANT: Based on the search intent above, focus on terms that match this specific need. {search_intent}' if search_intent else ''}

Format each suggestion as:
TERM: actual search term
RELEVANCE: 0.1-1.0 (how relevant)
COVERAGE: 0.1-1.0 (unique businesses found)
EFFICIENCY: 0.1-1.0 (results per credit)
REASON: why this term is valuable

Example:
TERM: roofing companies
RELEVANCE: 0.9
COVERAGE: 0.3
EFFICIENCY: 0.8
REASON: Common business listing term"""

        return prompt
    
    def _parse_ai_suggestions(self, ai_response: str, original_vertical: str) -> List[VerticalSuggestion]:
        """Parse AI response into VerticalSuggestion objects."""
        suggestions = []
        lines = ai_response.strip().split('\n')
        
        current_suggestion = {}
        
        # Debug logging
        logger.debug(f"Parsing AI response: {ai_response}")
        
        for line in lines:
            line = line.strip()
            # Handle numbered items (e.g., "1. TERM: ..." or just "TERM: ...")
            if 'TERM:' in line:
                if current_suggestion and current_suggestion.get('term'):
                    suggestions.append(self._create_suggestion_from_dict(current_suggestion))
                # Extract term after TERM:
                term_start = line.find('TERM:') + 5
                current_suggestion = {'term': line[term_start:].strip()}
            elif 'RELEVANCE:' in line:
                try:
                    rel_start = line.find('RELEVANCE:') + 10
                    current_suggestion['relevance'] = float(line[rel_start:].strip())
                except:
                    current_suggestion['relevance'] = 0.8
            elif 'COVERAGE:' in line:
                try:
                    cov_start = line.find('COVERAGE:') + 9
                    current_suggestion['coverage'] = float(line[cov_start:].strip())
                except:
                    current_suggestion['coverage'] = 0.3
            elif 'EFFICIENCY:' in line:
                try:
                    eff_start = line.find('EFFICIENCY:') + 11
                    current_suggestion['efficiency'] = float(line[eff_start:].strip())
                except:
                    current_suggestion['efficiency'] = 0.7
            elif 'REASON:' in line:
                reason_start = line.find('REASON:') + 7
                current_suggestion['reasoning'] = line[reason_start:].strip()
        
        # Add last suggestion
        if current_suggestion and current_suggestion.get('term'):
            suggestions.append(self._create_suggestion_from_dict(current_suggestion))
        
        # Fallback if parsing failed or no valid suggestions
        if not suggestions or all(not s.term.strip() for s in suggestions):
            logger.warning(f"AI response parsing failed for '{original_vertical}', using fallback")
            return self._get_fallback_suggestions(original_vertical, "balanced")
        
        # Filter out empty terms
        suggestions = [s for s in suggestions if s.term.strip()]
        
        return suggestions
    
    def _create_suggestion_from_dict(self, suggestion_dict: Dict) -> VerticalSuggestion:
        """Create VerticalSuggestion from parsed dictionary."""
        return VerticalSuggestion(
            term=suggestion_dict.get('term', ''),
            relevance_score=suggestion_dict.get('relevance', 0.8),
            unique_coverage_estimate=suggestion_dict.get('coverage', 0.3),
            cost_efficiency_score=suggestion_dict.get('efficiency', 0.7),
            reasoning=suggestion_dict.get('reasoning', 'AI suggested term')
        )
    
    def _get_fallback_suggestions(self, vertical: str, mode: str) -> List[VerticalSuggestion]:
        """Get fallback suggestions using industry knowledge base."""
        if vertical in self.INDUSTRY_EXPANSIONS:
            base_terms = self.INDUSTRY_EXPANSIONS[vertical]
        else:
            # Generic expansion for unknown verticals
            base_terms = [
                f"{vertical} companies",
                f"{vertical} services", 
                f"{vertical} contractors",
                f"{vertical} professionals",
                f"{vertical} specialists"
            ]
        
        # Select terms based on mode
        mode_limits = {
            "minimal": 3,
            "balanced": 5,
            "comprehensive": 8
        }
        
        limit = mode_limits.get(mode, 5)
        selected_terms = base_terms[:limit]
        
        suggestions = []
        for i, term in enumerate(selected_terms):
            # Assign scores based on position (first terms are usually better)
            relevance = max(0.9 - (i * 0.1), 0.6)
            coverage = max(0.4 - (i * 0.05), 0.15)
            efficiency = max(0.8 - (i * 0.08), 0.5)
            
            suggestions.append(VerticalSuggestion(
                term=term,
                relevance_score=relevance,
                unique_coverage_estimate=coverage,
                cost_efficiency_score=efficiency,
                reasoning=f"Industry knowledge base suggestion #{i+1}"
            ))
        
        return suggestions
    
    def _apply_cost_constraints(self, 
                               suggestions: List[VerticalSuggestion],
                               location_intel: Dict,
                               priority: str,
                               max_budget: Optional[int],
                               max_multiplier: Optional[float]) -> List[str]:
        """Apply cost constraints to select optimal terms using contribution data."""
        if not suggestions:
            return []
        
        base_cost = location_intel['cost_per_term']
        location_type = location_intel.get('location_type', 'unknown')
        
        # Calculate constraints
        if max_budget:
            max_terms_budget = max_budget // base_cost
        else:
            max_terms_budget = float('inf')
        
        if max_multiplier:
            max_terms_multiplier = int(max_multiplier)
        else:
            max_terms_multiplier = float('inf')
        
        # Enhance suggestions with historical contribution data
        enhanced_suggestions = []
        for suggestion in suggestions:
            enhanced_score = suggestion.cost_efficiency_score
            
            # Get historical performance if available
            if self.contribution_analyzer:
                performance_score = self.contribution_analyzer.get_term_performance_score(
                    suggestion.term, location_type
                )
                # Blend AI score with historical data (70% historical, 30% AI)
                enhanced_score = performance_score * 0.7 + suggestion.cost_efficiency_score * 0.3
                
                # Log enhanced scoring for debugging
                logger.debug(f"Term '{suggestion.term}': AI score {suggestion.cost_efficiency_score:.2f}, "
                           f"historical score {performance_score:.2f}, enhanced score {enhanced_score:.2f}")
            
            # Create enhanced suggestion
            enhanced_suggestions.append(VerticalSuggestion(
                term=suggestion.term,
                relevance_score=suggestion.relevance_score,
                unique_coverage_estimate=suggestion.unique_coverage_estimate,
                cost_efficiency_score=enhanced_score,
                reasoning=suggestion.reasoning + (
                    f" (enhanced with historical data)" if self.contribution_analyzer else ""
                )
            ))
        
        # Determine selection strategy based on priority
        if priority == ExpansionPriority.COST_CONTROL.value:
            # Minimize cost - select highest efficiency terms
            sorted_suggestions = sorted(enhanced_suggestions, key=lambda s: s.cost_efficiency_score, reverse=True)
            max_terms = min(max_terms_budget, max_terms_multiplier, 3)  # Cap at 3 for cost control
        elif priority == ExpansionPriority.COVERAGE.value:
            # Maximize coverage - select highest coverage terms
            sorted_suggestions = sorted(enhanced_suggestions, key=lambda s: s.unique_coverage_estimate, reverse=True)
            max_terms = min(max_terms_budget, max_terms_multiplier, len(enhanced_suggestions))
        else:
            # Balanced - optimize relevance * efficiency score
            sorted_suggestions = sorted(
                enhanced_suggestions, 
                key=lambda s: s.relevance_score * s.cost_efficiency_score, 
                reverse=True
            )
            max_terms = min(max_terms_budget, max_terms_multiplier, 5)  # Reasonable default
        
        # Filter out terms with high predicted overlap
        selected_terms = []
        for suggestion in sorted_suggestions:
            if len(selected_terms) >= max_terms:
                break
            
            # Check overlap with already selected terms
            should_add = True
            if self.contribution_analyzer and selected_terms:
                for selected_term in selected_terms:
                    overlap_rate = self.contribution_analyzer.predict_term_overlap(
                        suggestion.term, selected_term, location_type
                    )
                    # Skip if high overlap (>60%) with already selected term
                    if overlap_rate > 0.6:
                        logger.debug(f"Skipping '{suggestion.term}' due to {overlap_rate:.1%} overlap with '{selected_term}'")
                        should_add = False
                        break
            
            if should_add:
                selected_terms.append(suggestion.term)
        
        # Ensure at least 1 term, but respect hard budget constraints
        if not selected_terms and sorted_suggestions:
            selected_terms = [sorted_suggestions[0].term]
        
        logger.info(f"Selected {len(selected_terms)} terms with contribution-enhanced scoring")
        return selected_terms
    
    def _analyze_costs(self, selected_terms: List[str], location_intel: Dict, original_vertical: str) -> Dict:
        """Analyze cost implications of expansion."""
        base_cost = location_intel['cost_per_term']
        expanded_cost = len(selected_terms) * base_cost
        
        return {
            "original_terms": 1,
            "expanded_terms": len(selected_terms),
            "original_estimated_credits": base_cost,
            "expanded_estimated_credits": expanded_cost,
            "original_estimated_cost_usd": base_cost * 0.001,
            "expanded_estimated_cost_usd": expanded_cost * 0.001,
            "cost_multiplier": expanded_cost / base_cost if base_cost > 0 else 1,
            "credits_per_term": base_cost,
            "location_type": location_intel['location_type'],
            "pages_per_term": location_intel['max_pages']
        }
    
    def _analyze_coverage(self, original: str, selected: List[str], all_suggestions: List[VerticalSuggestion]) -> Dict:
        """Analyze coverage implications of expansion."""
        if not selected:
            return {"estimated_coverage_increase": "0%", "reasoning": "No expansion performed"}
        
        # Calculate coverage estimates
        original_coverage = 0.3  # Assume broad term gets 30% of businesses
        
        # Sum unique coverage from selected terms
        selected_suggestions = [s for s in all_suggestions if s.term in selected]
        expanded_coverage = min(0.95, original_coverage + sum(s.unique_coverage_estimate for s in selected_suggestions))
        
        coverage_increase = (expanded_coverage - original_coverage) / original_coverage
        
        return {
            "original_estimated_coverage": f"{original_coverage:.0%}",
            "expanded_estimated_coverage": f"{expanded_coverage:.0%}",
            "estimated_coverage_increase": f"{coverage_increase:.0%}",
            "reasoning": f"Expansion from '{original}' to {len(selected)} terms increases business coverage"
        }
    
    def _calculate_base_cost(self, location_intel: Dict) -> int:
        """Calculate base cost for a single term."""
        return location_intel['cost_per_term']
    
    def _estimate_results_for_terms(self, terms: List[str], location_intel: Dict) -> int:
        """Estimate total results for given terms."""
        # Simple estimation based on location density
        density_multipliers = {
            'dense_urban': 20,
            'urban': 15,
            'suburban': 10,
            'rural': 6,
            'unknown': 12
        }
        
        multiplier = density_multipliers.get(location_intel['business_density'], 12)
        pages_per_term = location_intel['max_pages']
        
        # Estimate with some overlap reduction
        base_results = len(terms) * pages_per_term * multiplier
        overlap_reduction = min(0.3, (len(terms) - 1) * 0.1)  # 10% overlap per additional term
        
        return int(base_results * (1 - overlap_reduction))
    
    def _generate_expansion_reasoning(self, original: str, selected: List[str], priority: str, cost_analysis: Dict) -> str:
        """Generate human-readable reasoning for expansion decisions."""
        if len(selected) <= 1:
            return f"No expansion needed or budget too constrained for '{original}'"
        
        priority_desc = {
            "cost_control": "cost-conscious",
            "balanced": "balanced cost/coverage",
            "coverage": "coverage-focused"
        }
        
        desc = priority_desc.get(priority, "balanced")
        cost_mult = cost_analysis["cost_multiplier"]
        
        return (f"Expanded '{original}' to {len(selected)} terms using {desc} strategy. "
               f"Cost increase: {cost_mult:.1f}x for comprehensive coverage of business variations.")
    
    def _create_no_expansion_result(self, vertical: str, start_time: float) -> ExpansionResult:
        """Create result when no expansion is needed."""
        response_time = int((time.time() - start_time) * 1000)
        
        return ExpansionResult(
            original_vertical=vertical,
            suggested_verticals=[],
            selected_verticals=[vertical],
            expansion_reasoning=f"'{vertical}' is already specific, no expansion needed",
            cost_analysis={
                "original_terms": 1,
                "expanded_terms": 1, 
                "cost_multiplier": 1.0,
                "no_expansion": True
            },
            coverage_analysis={
                "estimated_coverage_increase": "0%",
                "reasoning": "No expansion performed"
            },
            location_context=None,
            cache_hit=False,
            response_time_ms=response_time,
            ai_enabled=self.enabled
        )
    
    def _generate_cache_key(self, vertical: str, location: Optional[str], mode: str, priority: str) -> str:
        """Generate cache key for expansion request."""
        cache_data = {
            'vertical': vertical,
            'location': location or '',
            'mode': mode,
            'priority': priority
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_expansion(self, cache_key: str) -> Optional[ExpansionResult]:
        """Get cached expansion result."""
        # Placeholder for cache implementation
        return None
    
    def _cache_expansion(self, cache_key: str, result: ExpansionResult):
        """Cache expansion result."""
        # Placeholder for cache implementation
        pass
    
    def get_stats(self) -> Dict:
        """Get expansion service statistics."""
        return {
            'enabled': self.enabled,
            'groq_available': GROQ_AVAILABLE,
            'has_api_key': bool(settings.GROQ_API_KEY),
            'has_geo_optimizer': self.geo_optimizer is not None,
            'cache_enabled': self.cache_manager is not None,
            'industry_categories': len(self.INDUSTRY_EXPANSIONS)
        }