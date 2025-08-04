import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from utils.geographic import GeographicOptimizer, LocationCluster
from services.database import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class SearchPrediction:
    """Prediction of search costs and results."""
    estimated_credits: int
    estimated_cost: float
    estimated_results: int
    estimated_unique_results: int
    estimated_duplicate_rate: float
    optimization_savings: Dict
    recommendations: List[str]
    risk_level: str  # 'low', 'medium', 'high'

class SearchOptimizer:
    """Provides pre-search optimization and cost prediction."""
    
    # Base estimates for search results
    BASE_RESULTS_PER_PAGE = 20
    CREDITS_PER_PAGE = 3
    
    # Location type result estimates (results per page average)
    LOCATION_TYPE_ESTIMATES = {
        'dense_urban': 18,     # High business density
        'urban': 15,           # Medium-high density
        'suburban': 12,        # Medium density
        'rural': 8,            # Low density
        'unknown': 12          # Conservative estimate
    }
    
    # Vertical complexity multipliers
    VERTICAL_COMPLEXITY = {
        'simple': 1.0,         # "pizza", "coffee"
        'complex': 0.7,        # "neapolitan pizza restaurant"
        'generic': 1.3,        # "restaurant", "business"
        'niche': 0.5          # "vegan gluten-free bakery"
    }
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.geo_optimizer = GeographicOptimizer()
    
    def analyze_search_request(self, verticals: List[str], locations: List[Tuple[str, float, float]], 
                             zoom: int, max_pages: Optional[int]) -> SearchPrediction:
        """
        Analyze a search request and provide cost predictions and optimizations.
        
        Args:
            verticals: List of business types to search
            locations: List of (name, latitude, longitude) tuples
            zoom: Map zoom level
            max_pages: Maximum number of result pages
            
        Returns:
            SearchPrediction with cost estimates and recommendations
        """
        logger.info(f"Analyzing search request: {len(verticals)} verticals, {len(locations)} locations")
        
        # Step 1: Optimize locations using clustering
        location_clusters = self.geo_optimizer.cluster_locations(locations)
        
        # Step 2: Estimate results for each cluster
        cluster_estimates = []
        total_estimated_credits = 0
        total_estimated_results = 0
        
        # Use default max_pages if not specified
        effective_max_pages = max_pages or 3  # Default to 3 if None
        
        for cluster in location_clusters:
            cluster_estimate = self._estimate_cluster_results(cluster, verticals, effective_max_pages)
            cluster_estimates.append(cluster_estimate)
            total_estimated_credits += cluster_estimate['credits']
            total_estimated_results += cluster_estimate['results']
        
        # Step 3: Calculate duplicate rate estimate
        estimated_duplicate_rate = self._estimate_duplicate_rate(location_clusters, verticals)
        estimated_unique_results = int(total_estimated_results * (1 - estimated_duplicate_rate))
        
        # Step 4: Calculate optimization savings  
        optimization_savings = self._calculate_predicted_savings(
            len(locations), len(location_clusters), total_estimated_credits, len(verticals)
        )
        
        # Step 5: Generate recommendations
        recommendations = self._generate_pre_search_recommendations(
            verticals, location_clusters, total_estimated_credits, estimated_duplicate_rate
        )
        
        # Step 6: Assess risk level
        risk_level = self._assess_search_risk(
            total_estimated_credits, estimated_unique_results, estimated_duplicate_rate
        )
        
        prediction = SearchPrediction(
            estimated_credits=total_estimated_credits,
            estimated_cost=total_estimated_credits * 0.001,
            estimated_results=total_estimated_results,
            estimated_unique_results=estimated_unique_results,
            estimated_duplicate_rate=estimated_duplicate_rate,
            optimization_savings=optimization_savings,
            recommendations=recommendations,
            risk_level=risk_level
        )
        
        # Log prediction summary
        self._log_prediction_summary(prediction, location_clusters)
        
        return prediction
    
    def _estimate_cluster_results(self, cluster: LocationCluster, verticals: List[str], 
                                max_pages: int) -> Dict:
        """Estimate results for a single location cluster."""
        # Get base results estimate for location type
        base_results_per_page = self.LOCATION_TYPE_ESTIMATES[cluster.location_type.value]
        
        # Adjust for vertical complexity
        vertical_multiplier = self._calculate_vertical_complexity(verticals)
        results_per_page = int(base_results_per_page * vertical_multiplier)
        
        # Check historical data if available
        historical_data = self._get_historical_performance(cluster)
        if historical_data:
            results_per_page = int((results_per_page + historical_data['avg_results']) / 2)
        
        # Estimate pagination effectiveness (diminishing returns)
        total_results = 0
        total_credits = 0
        
        for page in range(1, max_pages + 1):
            if page == 1:
                page_results = results_per_page
            else:
                # Diminishing returns: each page has fewer unique results
                page_results = int(results_per_page * (0.7 ** (page - 1)))
                
            if page_results < 5:  # Stop if too few results expected
                break
                
            total_results += page_results * len(verticals)
            total_credits += self.CREDITS_PER_PAGE * len(verticals)
        
        return {
            'cluster_name': cluster.centroid_name,
            'location_type': cluster.location_type.value,
            'results': total_results,
            'credits': total_credits,
            'pages_estimated': min(max_pages, total_credits // (self.CREDITS_PER_PAGE * len(verticals)))
        }
    
    def _calculate_vertical_complexity(self, verticals: List[str]) -> float:
        """Calculate a complexity multiplier based on vertical terms."""
        total_multiplier = 0
        
        for vertical in verticals:
            words = vertical.lower().split()
            
            if len(words) == 1:
                # Single word - check if generic or specific
                if vertical.lower() in ['restaurant', 'business', 'store', 'shop']:
                    total_multiplier += self.VERTICAL_COMPLEXITY['generic']
                else:
                    total_multiplier += self.VERTICAL_COMPLEXITY['simple']
            elif len(words) >= 3:
                # Complex/specific terms
                if any(word in vertical.lower() for word in ['vegan', 'gluten-free', 'organic', 'artisan']):
                    total_multiplier += self.VERTICAL_COMPLEXITY['niche']
                else:
                    total_multiplier += self.VERTICAL_COMPLEXITY['complex']
            else:
                # Two words - moderate complexity
                total_multiplier += self.VERTICAL_COMPLEXITY['simple']
        
        return total_multiplier / len(verticals)  # Average multiplier
    
    def _estimate_duplicate_rate(self, location_clusters: List[LocationCluster], 
                                verticals: List[str]) -> float:
        """Estimate the duplicate rate based on cluster proximity and verticals."""
        if len(location_clusters) <= 1:
            return 0.1  # Single cluster has minimal duplicates
        
        # Calculate average distance between clusters
        total_distance = 0
        comparisons = 0
        
        for i, cluster1 in enumerate(location_clusters):
            for cluster2 in location_clusters[i+1:]:
                distance = self.geo_optimizer.haversine_distance(
                    cluster1.centroid_lat, cluster1.centroid_lon,
                    cluster2.centroid_lat, cluster2.centroid_lon
                )
                total_distance += distance
                comparisons += 1
        
        avg_distance = total_distance / max(comparisons, 1)
        
        # Estimate duplicate rate based on distance
        if avg_distance >= 50:        # Far apart
            base_duplicate_rate = 0.05
        elif avg_distance >= 25:      # Moderate distance
            base_duplicate_rate = 0.15
        elif avg_distance >= 15:      # Close
            base_duplicate_rate = 0.30
        else:                         # Very close
            base_duplicate_rate = 0.50
        
        # Adjust for vertical specificity
        generic_verticals = sum(1 for v in verticals if v.lower() in ['restaurant', 'business', 'store'])
        if generic_verticals > 0:
            base_duplicate_rate += 0.1 * (generic_verticals / len(verticals))
        
        return min(base_duplicate_rate, 0.8)  # Cap at 80%
    
    def _get_historical_performance(self, cluster: LocationCluster) -> Optional[Dict]:
        """Get historical search performance for similar locations."""
        # Check if we have historical data for locations in this cluster
        for location in cluster.locations:
            intelligence = self.geo_optimizer.location_intelligence.get(location[0])
            if intelligence and intelligence['searches'] > 0:
                return {
                    'avg_results': intelligence['avg_results_per_page'],
                    'searches': intelligence['searches']
                }
        return None
    
    def _calculate_predicted_savings(self, original_locations: int, optimized_clusters: int, 
                                   estimated_credits: int, verticals_count: int) -> Dict:
        """Calculate predicted savings from optimization."""
        # Estimate what the cost would be without optimization
        estimated_original_credits = original_locations * verticals_count * 9  # Assume 3 pages avg
        
        credits_saved = max(0, estimated_original_credits - estimated_credits)
        savings_percentage = (credits_saved / max(estimated_original_credits, 1)) * 100
        
        return {
            'estimated_original_credits': estimated_original_credits,
            'optimized_credits': estimated_credits,
            'credits_saved': credits_saved,
            'savings_percentage': savings_percentage,
            'locations_reduced': original_locations - optimized_clusters
        }
    
    def _generate_pre_search_recommendations(self, verticals: List[str], 
                                           location_clusters: List[LocationCluster],
                                           estimated_credits: int,
                                           estimated_duplicate_rate: float) -> List[str]:
        """Generate optimization recommendations before search execution."""
        recommendations = []
        
        # Cost efficiency recommendations
        estimated_cost = estimated_credits * 0.001
        if estimated_cost > 0.05:  # > $0.05
            recommendations.append(f"💰 Estimated cost is ${estimated_cost:.3f} - consider reducing locations or pages")
        
        # Duplicate rate recommendations
        if estimated_duplicate_rate > 0.4:
            recommendations.append(f"🔄 High duplicate rate expected ({estimated_duplicate_rate:.1%}) - locations may be too close")
            recommendations.append("📍 Consider increasing location clustering distance or reducing location count")
        
        # Vertical optimization
        generic_count = sum(1 for v in verticals if v.lower() in ['restaurant', 'business', 'store', 'shop'])
        if generic_count > 0:
            recommendations.append("🎯 Generic verticals detected - consider more specific terms to reduce irrelevant results")
        
        # Location clustering feedback
        if len(location_clusters) < len([loc for cluster in location_clusters for loc in cluster.locations]) * 0.7:
            recommendations.append("✨ Good location clustering achieved - optimization will help reduce costs")
        
        # Zoom level suggestions
        dense_clusters = [c for c in location_clusters if c.location_type.value == 'dense_urban']
        if len(dense_clusters) > 0 and any(c.recommended_zoom < 15 for c in dense_clusters):
            recommendations.append("🔍 Consider higher zoom levels for urban areas to reduce irrelevant results")
        
        if not recommendations:
            recommendations.append("✅ Search parameters look well-optimized")
        
        return recommendations
    
    def _assess_search_risk(self, estimated_credits: int, estimated_unique_results: int, 
                          estimated_duplicate_rate: float) -> str:
        """Assess the risk level of the search based on estimates."""
        risk_factors = 0
        
        # High cost risk
        if estimated_credits > 50:  # > $0.05
            risk_factors += 2
        elif estimated_credits > 30:  # > $0.03
            risk_factors += 1
        
        # Low results risk
        if estimated_unique_results < 20:
            risk_factors += 2
        elif estimated_unique_results < 50:
            risk_factors += 1
        
        # High duplicate risk
        if estimated_duplicate_rate > 0.6:
            risk_factors += 2
        elif estimated_duplicate_rate > 0.4:
            risk_factors += 1
        
        # Poor cost efficiency risk
        cost_per_result = (estimated_credits * 0.001) / max(estimated_unique_results, 1)
        if cost_per_result > 0.0005:  # > $0.0005 per result
            risk_factors += 2
        elif cost_per_result > 0.0003:  # > $0.0003 per result
            risk_factors += 1
        
        if risk_factors >= 4:
            return 'high'
        elif risk_factors >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _log_prediction_summary(self, prediction: SearchPrediction, 
                              location_clusters: List[LocationCluster]):
        """Log a summary of the search prediction."""
        logger.info("🔮 SEARCH PREDICTION SUMMARY:")
        logger.info(f"   Estimated cost: ${prediction.estimated_cost:.4f} ({prediction.estimated_credits} credits)")
        logger.info(f"   Expected results: {prediction.estimated_unique_results} unique ({prediction.estimated_results} total)")
        logger.info(f"   Duplicate rate: {prediction.estimated_duplicate_rate:.1%}")
        logger.info(f"   Risk level: {prediction.risk_level.upper()}")
        
        if prediction.optimization_savings['savings_percentage'] > 0:
            logger.info(f"   Optimization savings: {prediction.optimization_savings['savings_percentage']:.1f}%")
        
        logger.info(f"   Location clusters: {len(location_clusters)}")
        
        if prediction.recommendations:
            logger.info("   Recommendations:")
            for rec in prediction.recommendations:
                logger.info(f"     • {rec}")
    
    def should_proceed_with_search(self, prediction: SearchPrediction) -> Tuple[bool, str]:
        """
        Determine if a search should proceed based on predictions.
        
        Returns:
            Tuple of (should_proceed, reason)
        """
        if prediction.risk_level == 'high':
            return False, "High risk search - consider optimizing parameters first"
        
        if prediction.estimated_cost > 1.00:  # > $1.00 - raised threshold
            return False, f"Estimated cost (${prediction.estimated_cost:.3f}) exceeds safety threshold"
        
        if prediction.estimated_duplicate_rate > 0.7:
            return False, f"Very high duplicate rate expected ({prediction.estimated_duplicate_rate:.1%})"
        
        cost_per_result = prediction.estimated_cost / max(prediction.estimated_unique_results, 1)
        if cost_per_result > 0.001:  # > $0.001 per result
            return False, f"Poor cost efficiency predicted (${cost_per_result:.5f} per result)"
        
        return True, "Search parameters look good to proceed"