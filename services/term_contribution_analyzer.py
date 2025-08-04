"""
Term Contribution Analyzer for the Business Mapper API.

This service tracks and analyzes which search terms contribute unique, valuable results
vs duplicates across expanded vertical searches. It provides insights to improve AI
expansion suggestions and optimize search efficiency.
"""

import hashlib
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

from services.database import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class TermContribution:
    """Contribution data for a single search term."""
    term: str
    total_results: int
    unique_results: int
    overlap_results: int
    credits_used: int
    pages_searched: int
    contribution_rate: float
    cost_efficiency: float
    discovery_value: float
    overlaps_with: List[str]

@dataclass
class ContributionAnalysis:
    """Complete contribution analysis for a search session."""
    session_id: str
    location_context: str
    location_type: str
    total_terms: int
    total_results: int
    total_unique_results: int
    overall_duplicate_rate: float
    term_contributions: List[TermContribution]
    overlap_patterns: Dict[Tuple[str, str], float]
    recommendations: List[str]
    analysis_time_ms: int

class TermContributionAnalyzer:
    """
    Analyzes term contributions in expanded searches to track which terms
    provide unique value vs duplicates, enabling smarter AI expansion suggestions.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the term contribution analyzer.
        
        Args:
            db_manager: Database manager for storing contribution data
        """
        self.db = db_manager
        self.current_sessions = {}  # Active search sessions being tracked
        
    def start_session(self, terms: List[str], location_context: str, 
                     location_type: str) -> str:
        """
        Start a new contribution tracking session.
        
        Args:
            terms: List of search terms being analyzed
            location_context: Location context (e.g., "New York, NY")
            location_type: Type of location (dense_urban, urban, suburban, rural)
            
        Returns:
            Session ID for tracking this analysis
        """
        session_id = str(uuid.uuid4())
        
        self.current_sessions[session_id] = {
            'terms': terms,
            'location_context': location_context,
            'location_type': location_type,
            'term_results': {},  # term -> list of business IDs
            'term_metadata': {},  # term -> metadata (credits, pages, etc.)
            'start_time': time.time()
        }
        
        logger.info(f"Started contribution session {session_id[:8]}... for {len(terms)} terms in {location_context}")
        return session_id
    
    def record_term_results(self, session_id: str, term: str, results: List[Dict],
                          credits_used: int, pages_searched: int) -> bool:
        """
        Record results for a specific term in the session.
        
        Args:
            session_id: Session ID from start_session
            term: The search term
            results: List of business results from this term
            credits_used: Credits used for this term's search
            pages_searched: Number of pages searched
            
        Returns:
            True if recorded successfully, False otherwise
        """
        if session_id not in self.current_sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self.current_sessions[session_id]
        
        # Extract business identifiers for overlap analysis
        business_ids = []
        for result in results:
            # Use multiple identifiers to ensure robust matching
            identifier = self._generate_business_identifier(result)
            if identifier:
                business_ids.append(identifier)
        
        session['term_results'][term] = business_ids
        session['term_metadata'][term] = {
            'credits_used': credits_used,
            'pages_searched': pages_searched,
            'results_count': len(results)
        }
        
        logger.debug(f"Recorded {len(business_ids)} results for term '{term}' in session {session_id[:8]}...")
        return True
    
    def analyze_session(self, session_id: str) -> Optional[ContributionAnalysis]:
        """
        Analyze contribution patterns for a completed session.
        
        Args:
            session_id: Session ID to analyze
            
        Returns:
            ContributionAnalysis with detailed metrics and insights
        """
        if session_id not in self.current_sessions:
            logger.error(f"Session {session_id} not found")
            return None
        
        start_time = time.time()
        session = self.current_sessions[session_id]
        
        # Calculate overlap patterns between all term pairs
        overlap_patterns = self._calculate_overlap_patterns(session['term_results'])
        
        # Calculate individual term contributions
        term_contributions = []
        total_unique_results = set()
        
        for term in session['terms']:
            if term not in session['term_results']:
                logger.warning(f"No results recorded for term '{term}' in session")
                continue
            
            contribution = self._calculate_term_contribution(
                term, session, overlap_patterns
            )
            term_contributions.append(contribution)
            total_unique_results.update(session['term_results'][term])
        
        # Calculate overall metrics
        total_results = sum(len(session['term_results'].get(term, [])) for term in session['terms'])
        overall_duplicate_rate = 1 - (len(total_unique_results) / max(total_results, 1))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            term_contributions, overlap_patterns, overall_duplicate_rate
        )
        
        analysis_time_ms = int((time.time() - start_time) * 1000)
        
        # Create analysis result
        analysis = ContributionAnalysis(
            session_id=session_id,
            location_context=session['location_context'],
            location_type=session['location_type'],
            total_terms=len(session['terms']),
            total_results=total_results,
            total_unique_results=len(total_unique_results),
            overall_duplicate_rate=overall_duplicate_rate,
            term_contributions=term_contributions,
            overlap_patterns=overlap_patterns,
            recommendations=recommendations,
            analysis_time_ms=analysis_time_ms
        )
        
        # Store results in database
        self._store_analysis_results(analysis)
        
        # Clean up session
        del self.current_sessions[session_id]
        
        logger.info(f"Analyzed session {session_id[:8]}...: "
                   f"{len(term_contributions)} terms, "
                   f"{len(total_unique_results)} unique results, "
                   f"{overall_duplicate_rate:.1%} duplicate rate")
        
        return analysis
    
    def get_term_performance_score(self, term: str, location_type: str = None) -> float:
        """
        Get historical performance score for a term.
        
        Args:
            term: The search term
            location_type: Optional location type filter
            
        Returns:
            Performance score (0.0 to 1.0), where 1.0 is excellent performance
        """
        performance = self.db.get_term_performance(term, location_type)
        
        if not performance or performance.get('confidence_score', 0) < 0.3:
            # Insufficient data, return neutral score
            return 0.5
        
        # Calculate weighted score based on multiple metrics
        contribution_score = performance.get('avg_contribution_rate', 0.5)
        efficiency_score = min(1.0, performance.get('avg_cost_efficiency', 1.0) / 5.0)  # Normalize to 0-1
        discovery_score = performance.get('avg_discovery_value', 0.5)
        confidence = performance.get('confidence_score', 0.5)
        
        # Weighted average with confidence scaling
        weighted_score = (
            contribution_score * 0.4 +
            efficiency_score * 0.3 +
            discovery_score * 0.3
        ) * confidence
        
        return min(1.0, max(0.0, weighted_score))
    
    def predict_term_overlap(self, term1: str, term2: str, location_type: str = None) -> float:
        """
        Predict overlap rate between two terms based on historical data.
        
        Args:
            term1: First term
            term2: Second term
            location_type: Optional location type context
            
        Returns:
            Predicted overlap rate (0.0 to 1.0)
        """
        # Check direct historical overlap
        overlap_rate = self.db.get_term_overlap_rate(term1, term2, location_type)
        
        if overlap_rate > 0:
            return overlap_rate
        
        # Use semantic similarity heuristics for unknown pairs
        return self._estimate_semantic_overlap(term1, term2)
    
    def get_expansion_recommendations(self, base_term: str, candidate_terms: List[str],
                                   location_type: str = None, max_terms: int = 5) -> List[Tuple[str, float]]:
        """
        Get recommended terms for expansion based on contribution analysis.
        
        Args:
            base_term: The original search term
            candidate_terms: List of potential expansion terms
            location_type: Location type context
            max_terms: Maximum number of terms to recommend
            
        Returns:
            List of (term, score) tuples sorted by recommendation score
        """
        recommendations = []
        
        for term in candidate_terms:
            if term == base_term:
                continue
            
            # Get individual performance score
            performance_score = self.get_term_performance_score(term, location_type)
            
            # Get predicted overlap with base term
            overlap_rate = self.predict_term_overlap(base_term, term, location_type)
            
            # Calculate recommendation score
            # Higher performance and lower overlap = better recommendation
            uniqueness_bonus = 1.0 - overlap_rate
            recommendation_score = performance_score * 0.7 + uniqueness_bonus * 0.3
            
            recommendations.append((term, recommendation_score))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:max_terms]
    
    def _generate_business_identifier(self, result: Dict) -> Optional[str]:
        """
        Generate a unique identifier for a business result to track duplicates.
        
        Args:
            result: Business result dictionary
            
        Returns:
            Unique identifier string or None if insufficient data
        """
        # Try multiple identification strategies
        identifiers = []
        
        # Primary: Google Place ID (most reliable)
        place_id = result.get('placeId', '').strip()
        if place_id:
            identifiers.append(f"place_id:{place_id}")
        
        # Secondary: Name + normalized address
        name = result.get('name', '').strip().lower()
        address = result.get('address', '').strip().lower()
        if name and address:
            # Normalize address by removing common variations
            normalized_addr = address.replace(' street', ' st').replace(' avenue', ' ave')
            identifiers.append(f"name_addr:{name}:{normalized_addr}")
        
        # Tertiary: Phone number (if available)
        phone = result.get('phoneNumber', '').strip()
        if phone and name:
            # Clean phone number
            clean_phone = ''.join(c for c in phone if c.isdigit())
            if len(clean_phone) >= 10:
                identifiers.append(f"name_phone:{name}:{clean_phone}")
        
        # Quaternary: Website (if available)
        website = result.get('website', '').strip().lower()
        if website and name:
            # Normalize website
            website = website.replace('http://', '').replace('https://', '').replace('www.', '')
            identifiers.append(f"name_web:{name}:{website}")
        
        # Return the most reliable identifier available
        return identifiers[0] if identifiers else None
    
    def _calculate_overlap_patterns(self, term_results: Dict[str, List[str]]) -> Dict[Tuple[str, str], float]:
        """Calculate overlap rates between all term pairs."""
        overlap_patterns = {}
        terms = list(term_results.keys())
        
        for i, term1 in enumerate(terms):
            for term2 in terms[i+1:]:
                results1 = set(term_results.get(term1, []))
                results2 = set(term_results.get(term2, []))
                
                if not results1 or not results2:
                    overlap_rate = 0.0
                else:
                    intersection = len(results1 & results2)
                    union = len(results1 | results2)
                    overlap_rate = intersection / union if union > 0 else 0.0
                
                overlap_patterns[(term1, term2)] = overlap_rate
        
        return overlap_patterns
    
    def _calculate_term_contribution(self, term: str, session: Dict, 
                                   overlap_patterns: Dict) -> TermContribution:
        """Calculate contribution metrics for a single term."""
        term_results = set(session['term_results'].get(term, []))
        metadata = session['term_metadata'].get(term, {})
        
        # Calculate unique contributions
        all_other_results = set()
        overlaps_with = []
        
        for other_term, other_results in session['term_results'].items():
            if other_term != term:
                other_set = set(other_results)
                all_other_results.update(other_set)
                
                # Check if this term overlaps with the other
                overlap_rate = overlap_patterns.get((term, other_term)) or \
                             overlap_patterns.get((other_term, term))
                if overlap_rate and overlap_rate > 0.1:  # 10% overlap threshold
                    overlaps_with.append(other_term)
        
        unique_results = len(term_results - all_other_results)
        overlap_results = len(term_results & all_other_results)
        total_results = len(term_results)
        
        # Calculate metrics
        contribution_rate = unique_results / max(total_results, 1)
        cost_efficiency = unique_results / max(metadata.get('credits_used', 1), 1)
        discovery_value = unique_results / max(total_results - overlap_results, 1)
        
        return TermContribution(
            term=term,
            total_results=total_results,
            unique_results=unique_results,
            overlap_results=overlap_results,
            credits_used=metadata.get('credits_used', 0),
            pages_searched=metadata.get('pages_searched', 0),
            contribution_rate=contribution_rate,
            cost_efficiency=cost_efficiency,
            discovery_value=discovery_value,
            overlaps_with=overlaps_with
        )
    
    def _generate_recommendations(self, term_contributions: List[TermContribution],
                                overlap_patterns: Dict, overall_duplicate_rate: float) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Sort terms by performance metrics
        high_performers = [tc for tc in term_contributions if tc.contribution_rate > 0.3]
        low_performers = [tc for tc in term_contributions if tc.contribution_rate < 0.1]
        high_overlap_terms = [tc for tc in term_contributions if len(tc.overlaps_with) > 2]
        
        # Overall duplicate rate recommendations
        if overall_duplicate_rate > 0.6:
            recommendations.append(f"High duplicate rate ({overall_duplicate_rate:.1%}) - consider reducing overlapping terms")
        elif overall_duplicate_rate < 0.2:
            recommendations.append(f"Low duplicate rate ({overall_duplicate_rate:.1%}) - terms are well-differentiated")
        
        # High performer recommendations
        if high_performers:
            best_performer = max(high_performers, key=lambda x: x.contribution_rate)
            recommendations.append(f"'{best_performer.term}' is your best contributor ({best_performer.contribution_rate:.1%} unique results)")
        
        # Low performer recommendations
        if low_performers:
            worst_performer = min(low_performers, key=lambda x: x.contribution_rate)
            recommendations.append(f"Consider removing '{worst_performer.term}' - only {worst_performer.contribution_rate:.1%} unique contribution")
        
        # High overlap recommendations
        if high_overlap_terms:
            for term_contrib in high_overlap_terms:
                recommendations.append(f"'{term_contrib.term}' overlaps heavily with {len(term_contrib.overlaps_with)} other terms")
        
        # Cost efficiency recommendations
        if term_contributions:
            most_efficient = max(term_contributions, key=lambda x: x.cost_efficiency)
            least_efficient = min(term_contributions, key=lambda x: x.cost_efficiency)
            
            if most_efficient.cost_efficiency > 3.0:
                recommendations.append(f"'{most_efficient.term}' is very cost-efficient ({most_efficient.cost_efficiency:.1f} results/credit)")
            
            if least_efficient.cost_efficiency < 0.5:
                recommendations.append(f"'{least_efficient.term}' is cost-inefficient ({least_efficient.cost_efficiency:.1f} results/credit)")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _store_analysis_results(self, analysis: ContributionAnalysis):
        """Store analysis results in the database."""
        try:
            # Store individual term contributions
            for term_contrib in analysis.term_contributions:
                self.db.record_term_contribution(
                    session_id=analysis.session_id,
                    term=term_contrib.term,
                    location_context=analysis.location_context,
                    location_type=analysis.location_type,
                    total_results=term_contrib.total_results,
                    unique_results=term_contrib.unique_results,
                    overlap_results=term_contrib.overlap_results,
                    credits_used=term_contrib.credits_used,
                    pages_searched=term_contrib.pages_searched,
                    metadata={
                        'overlaps_with': term_contrib.overlaps_with,
                        'session_total_terms': analysis.total_terms,
                        'session_duplicate_rate': analysis.overall_duplicate_rate
                    }
                )
                
                # Update performance history
                self.db.update_term_performance_history(term_contrib.term, analysis.location_type)
            
            # Store overlap patterns
            for (term1, term2), overlap_rate in analysis.overlap_patterns.items():
                self.db.record_term_overlap(term1, term2, analysis.location_type, overlap_rate)
            
            logger.debug(f"Stored analysis results for session {analysis.session_id[:8]}...")
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {str(e)}")
    
    def _estimate_semantic_overlap(self, term1: str, term2: str) -> float:
        """
        Estimate semantic overlap between two terms using simple heuristics.
        
        This is a fallback when no historical data is available.
        """
        term1_words = set(term1.lower().split())
        term2_words = set(term2.lower().split())
        
        # Calculate word overlap
        intersection = len(term1_words & term2_words)
        union = len(term1_words | term2_words)
        word_overlap = intersection / max(union, 1)
        
        # Apply business category heuristics
        business_categories = {
            'food': ['restaurant', 'cafe', 'diner', 'eatery', 'food', 'kitchen'],
            'automotive': ['auto', 'car', 'vehicle', 'automotive', 'garage', 'mechanic'],
            'health': ['doctor', 'clinic', 'medical', 'health', 'dentist', 'pharmacy'],
            'retail': ['store', 'shop', 'retail', 'boutique', 'market'],
            'service': ['service', 'company', 'business', 'professional']
        }
        
        # Check if terms belong to same category
        category_bonus = 0.0
        for category, keywords in business_categories.items():
            term1_in_cat = any(keyword in term1.lower() for keyword in keywords)
            term2_in_cat = any(keyword in term2.lower() for keyword in keywords)
            if term1_in_cat and term2_in_cat:
                category_bonus = 0.3
                break
        
        # Estimate overlap (conservative approach)
        estimated_overlap = min(0.8, word_overlap + category_bonus)
        
        return estimated_overlap
    
    def get_session_stats(self) -> Dict:
        """Get statistics about active and completed sessions."""
        return {
            'active_sessions': len(self.current_sessions),
            'active_session_ids': list(self.current_sessions.keys())
        }