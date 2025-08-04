import http.client
import json
import logging
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from requests.adapters import HTTPAdapter
from requests.sessions import Session
from urllib3.util.retry import Retry

from models.schemas import PlaceResult
from config.settings import settings
from utils.geographic import GeographicOptimizer, LocationCluster
from utils.address_parser import us_address_parser
from services.ai_validator import AIVerticalValidator
from services.validation_cache import ValidationCache
from services.term_contribution_analyzer import TermContributionAnalyzer
from services.city_classifier import AICityClassifier
from services.search_cache import SearchCache

logger = logging.getLogger(__name__)

class SerperMapsService:
    """Service for interacting with Serper Maps API with geographic optimization."""
    
    # Cost efficiency thresholds
    MIN_UNIQUE_RESULTS_PER_PAGE = 5  # Stop pagination if fewer unique results
    MAX_DUPLICATE_RATE = 0.5         # Stop if duplicate rate exceeds 50%
    TARGET_COST_PER_RESULT = 0.0002  # Target cost efficiency
    
    def __init__(self, api_key: str = None, db_manager=None):
        self.api_key = api_key or settings.SERPER_API_KEY
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is required")
        
        self.base_url = "google.serper.dev"
        self.endpoint = "/maps"
        
        # Initialize session with connection pooling for reuse
        self.session = Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,  # Number of connection pools to cache
            pool_maxsize=10,      # Maximum number of connections to save in the pool
            max_retries=retry_strategy
        )
        
        # Mount adapter for both http and https
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        })
        
        # Initialize AI validation with caching
        try:
            self.validation_cache = ValidationCache()
            self.ai_validator = AIVerticalValidator(cache_manager=self.validation_cache)
            logger.info("🤖 AI-powered vertical validation enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize AI validation: {str(e)}")
            self.validation_cache = None
            self.ai_validator = None
        
        # Initialize GeographicOptimizer with cache for AI city classification
        self.geo_optimizer = GeographicOptimizer(db_manager=self.validation_cache)
        
        # Initialize term contribution analyzer if database available
        self.db_manager = db_manager
        if db_manager:
            self.contribution_analyzer = TermContributionAnalyzer(db_manager)
            logger.info("📊 Term contribution tracking enabled")
        else:
            self.contribution_analyzer = None
        
        # Initialize AI city classifier for population type classification
        try:
            self.city_classifier = AICityClassifier(cache_manager=self.validation_cache)
            logger.info("🌍 AI city classification enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize AI city classifier: {str(e)}")
            self.city_classifier = None
        
        # Initialize search result cache
        self.search_cache = SearchCache(
            ttl_seconds=settings.SEARCH_CACHE_TTL if hasattr(settings, 'SEARCH_CACHE_TTL') else 3600,
            max_size=1000
        )
        logger.info("💾 Search result caching enabled")
    
    def search(self, verticals: List[str], locations: List[Tuple[str, float, float]], 
               zoom: int = 14, max_pages: Optional[int] = None, track_contributions: bool = True) -> Dict:
        """
        Search for businesses with geographic optimization and intelligent auto-pagination.
        
        Args:
            verticals: List of business types to search
            locations: List of (name, latitude, longitude) tuples
            zoom: Map zoom level (will be optimized per cluster)
            max_pages: Maximum number of result pages (auto-determined if None)
            track_contributions: Whether to track term contributions (requires expanded verticals)
            
        Returns:
            Dictionary with results and optimization metadata
        """
        # Step 1: Cluster locations to minimize overlap
        location_clusters = self.geo_optimizer.cluster_locations(locations)
        
        logger.info(f"Optimized {len(locations)} locations into {len(location_clusters)} clusters")
        
        # Initialize contribution tracking if enabled and multiple terms
        contribution_session_id = None
        if (track_contributions and self.contribution_analyzer and 
            len(verticals) > 1 and location_clusters):
            
            # Use the first cluster for location context
            primary_cluster = location_clusters[0]
            location_context = primary_cluster.centroid_name
            location_type = primary_cluster.location_type.value
            
            contribution_session_id = self.contribution_analyzer.start_session(
                verticals, location_context, location_type
            )
        
        # Step 2: Calculate intelligent pagination strategy
        pagination_strategy = self.geo_optimizer.calculate_smart_pagination(
            location_clusters, max_pages
        )
        
        if max_pages is None:
            logger.info("🤖 Using intelligent auto-pagination based on location types")
        else:
            logger.info(f"👤 User specified max_pages: {max_pages} (will be capped by location type limits)")
        
        # Log cluster analysis with pagination decisions
        for i, cluster in enumerate(location_clusters):
            strategy = pagination_strategy[f"cluster_{i}"]
            logger.info(f"Cluster {i+1}: {cluster.location_count} locations, "
                       f"centroid ({cluster.centroid_lat:.4f}, {cluster.centroid_lon:.4f}), "
                       f"zoom {cluster.recommended_zoom}, type {cluster.location_type.value}")
            logger.info(f"  └─ Pagination: {strategy['min_pages']}-{strategy['max_pages']} pages "
                       f"({'auto' if strategy['auto_pagination'] else 'manual'})")
        
        # Step 3: Execute optimized search with smart pagination
        results = []
        total_credits_used = 0
        cost_efficiency_data = []
        cluster_results = []
        vertical_results = {vertical: [] for vertical in verticals}  # Track results by vertical
        
        try:
            for cluster_idx, cluster in enumerate(location_clusters):
                # Get pagination strategy for this cluster
                strategy = pagination_strategy[f"cluster_{cluster_idx}"]
                cluster_max_pages = strategy['max_pages']
                
                cluster_data = self._search_cluster_with_smart_pagination(
                    verticals, cluster, strategy, cluster_idx + 1, contribution_session_id
                )
                
                results.extend(cluster_data['results'])
                total_credits_used += cluster_data['credits_used']
                cost_efficiency_data.append(cluster_data['efficiency'])
                
                # Collect results by vertical for contribution tracking
                if 'vertical_results' in cluster_data:
                    for vertical, vertical_cluster_results in cluster_data['vertical_results'].items():
                        vertical_results[vertical].extend(vertical_cluster_results)
                
                cluster_results.append({
                    'cluster_name': cluster.centroid_name,
                    'locations_count': cluster.location_count,
                    'results_found': len(cluster_data['results']),
                    'credits_used': cluster_data['credits_used'],
                    'pages_searched': cluster_data['pages_searched'],
                    'duplicate_rate': cluster_data['duplicate_rate'],
                    'cost_per_result': cluster_data['cost_per_result'],
                    'stopping_reason': cluster_data['stopping_reason']
                })
            
            # Step 3: Final deduplication across all clusters
            logger.info(f"Final deduplication: {len(results)} results from all clusters")
            unique_results = self._deduplicate_results(results)
            
            # Step 4: Calculate optimization metrics
            optimization_savings = self._calculate_optimization_savings(
                len(locations), len(location_clusters), total_credits_used, len(verticals)
            )
            
            # Step 5: Perform contribution analysis if tracking enabled
            contribution_analysis = None
            if contribution_session_id and self.contribution_analyzer:
                try:
                    # Record results for each vertical
                    vertical_credits = {}
                    vertical_pages = {}
                    
                    # Calculate per-vertical metrics (distribute costs equally)
                    credits_per_vertical = total_credits_used // len(verticals)
                    pages_per_vertical = sum(cr['pages_searched'] for cr in cluster_results) // len(verticals)
                    
                    for vertical in verticals:
                        vertical_credits[vertical] = credits_per_vertical
                        vertical_pages[vertical] = pages_per_vertical
                        
                        # Record results for this vertical
                        self.contribution_analyzer.record_term_results(
                            contribution_session_id,
                            vertical,
                            vertical_results.get(vertical, []),
                            vertical_credits[vertical],
                            vertical_pages[vertical]
                        )
                    
                    # Analyze the session
                    contribution_analysis = self.contribution_analyzer.analyze_session(contribution_session_id)
                    
                    if contribution_analysis:
                        logger.info(f"📊 Contribution analysis: {contribution_analysis.overall_duplicate_rate:.1%} "
                                   f"overall duplicate rate, {len(contribution_analysis.recommendations)} recommendations")
                    
                except Exception as e:
                    logger.error(f"Error in contribution analysis: {str(e)}")
            
            # Step 6: Update location intelligence
            for cluster in location_clusters:
                cluster_result_data = next(
                    (cr for cr in cluster_results if cr['cluster_name'] == cluster.centroid_name), 
                    None
                )
                if cluster_result_data:
                    for location in cluster.locations:
                        self.geo_optimizer.update_location_intelligence(
                            location[0], {
                                'total_results': cluster_result_data['results_found'],
                                'pages_searched': cluster_result_data['pages_searched']
                            }
                        )
            
            # Prepare response with contribution data
            response_data = {
                "results": list(unique_results.values()),
                "total_credits_used": total_credits_used,
                "cost_of_credits": total_credits_used * 0.001,
                "unique_businesses_found": len(unique_results),
                "raw_results_count": len(results),
                "search_metadata": {
                    "original_locations": len(locations),
                    "optimized_clusters": len(location_clusters),
                    "verticals_searched": len(verticals),
                    "optimization_savings": optimization_savings,
                    "cluster_results": cluster_results,
                    "pagination_strategy": pagination_strategy,
                    "avg_cost_per_result": (total_credits_used * 0.001) / max(len(unique_results), 1),
                    "duplicate_rate_overall": 1 - (len(unique_results) / max(len(results), 1)),
                    "auto_pagination_used": max_pages is None,
                    "cache_stats": self.get_cache_stats()
                }
            }
            
            # Add contribution analysis data if available
            if contribution_analysis:
                response_data["term_contribution_analysis"] = {
                    "session_id": contribution_analysis.session_id,
                    "overall_duplicate_rate": contribution_analysis.overall_duplicate_rate,
                    "total_unique_results": contribution_analysis.total_unique_results,
                    "term_contributions": [
                        {
                            "term": tc.term,
                            "total_results": tc.total_results,
                            "unique_results": tc.unique_results,
                            "contribution_rate": tc.contribution_rate,
                            "cost_efficiency": tc.cost_efficiency,
                            "discovery_value": tc.discovery_value,
                            "overlaps_with": tc.overlaps_with
                        }
                        for tc in contribution_analysis.term_contributions
                    ],
                    "recommendations": contribution_analysis.recommendations,
                    "analysis_time_ms": contribution_analysis.analysis_time_ms
                }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in optimized serper maps search: {str(e)}")
            raise
    
    def _search_cluster_with_smart_pagination(self, verticals: List[str], cluster: LocationCluster, 
                                             pagination_strategy: Dict, cluster_num: int, 
                                             contribution_session_id: str = None) -> Dict:
        """
        Search a single location cluster with intelligent pagination based on location type.
        
        Args:
            verticals: List of business types
            cluster: LocationCluster object
            pagination_strategy: Pagination rules for this cluster
            cluster_num: Cluster number for logging
            
        Returns:
            Dictionary with cluster search results and metrics
        """
        results = []
        credits_used = 0
        pages_searched = 0
        previous_unique_count = 0
        stopping_reason = None
        vertical_results = {vertical: [] for vertical in verticals}  # Track results by vertical
        
        # Extract pagination parameters
        min_pages = pagination_strategy['min_pages']
        max_pages = pagination_strategy['max_pages']
        stop_threshold = pagination_strategy['stop_threshold']
        duplicate_threshold = pagination_strategy['duplicate_threshold']
        
        logger.info(f"Searching cluster {cluster_num}: {cluster.centroid_name} "
                   f"(zoom {cluster.recommended_zoom}, type {cluster.location_type.value})")
        logger.info(f"  📄 Pagination strategy: {min_pages}-{max_pages} pages, "
                   f"stop at <{stop_threshold} unique or >{duplicate_threshold:.0%} duplicates")
        
        for page in range(1, max_pages + 1):
            # Create batch payload for this cluster and page, checking cache
            batch_payload, cached_results, cache_mapping = self._create_cluster_batch_payload(
                verticals, cluster, page
            )
            
            # Log cache usage
            if cached_results:
                logger.info(f"Cluster {cluster_num}, Page {page}: {len(batch_payload)} API queries, "
                           f"{len(cached_results)} from cache")
            else:
                logger.info(f"Cluster {cluster_num}, Page {page}: {len(batch_payload)} queries")
            
            # Execute batch request if we have uncached queries
            if batch_payload:
                batch_results = self._execute_batch_request(batch_payload)
            else:
                batch_results = []
            
            if not batch_results and not cached_results:
                logger.warning(f"No results returned for cluster {cluster_num}, page {page}")
                break
            
            # Process results for this page
            page_results = []
            page_credits = 0
            
            # First, process cached results
            for idx, data in enumerate(cached_results):
                if not isinstance(data, dict):
                    continue
                
                current_vertical = verticals[cache_mapping[idx]]
                
                # Process places from cache
                if 'places' in data:
                    for place in data['places']:
                        if self._matches_vertical(place, current_vertical):
                            processed = self._process_place(place, current_vertical)
                            if processed:
                                page_results.append(processed)
                                # Track by vertical for contribution analysis
                                vertical_results[current_vertical].append(processed)
                
                # No credits used for cached results
            
            # Then, process API results
            for idx, data in enumerate(batch_results):
                if not isinstance(data, dict):
                    continue
                
                # For API results, we need to determine which vertical this is for
                # Since cached results were removed, we need to account for the offset
                api_vertical_idx = idx
                for cache_idx in cache_mapping:
                    if cache_idx <= api_vertical_idx + len(cache_mapping):
                        api_vertical_idx += 1
                
                current_vertical = verticals[api_vertical_idx % len(verticals)]
                
                # Process places
                if 'places' in data:
                    for place in data['places']:
                        if self._matches_vertical(place, current_vertical):
                            processed = self._process_place(place, current_vertical)
                            if processed:
                                page_results.append(processed)
                                # Track by vertical for contribution analysis
                                vertical_results[current_vertical].append(processed)
                
                # Track credits for API results
                if 'credits' in data:
                    page_credits += data['credits']
                
                # Cache the API result for future use
                self.search_cache.set(
                    vertical=current_vertical,
                    location=cluster.centroid_name,
                    lat=cluster.centroid_lat,
                    lon=cluster.centroid_lon,
                    zoom=cluster.recommended_zoom,
                    page=page,
                    data=data
                )
            
            # Add page results and update counters
            results.extend(page_results)
            credits_used += page_credits
            pages_searched += 1
            
            # Check if we should continue pagination
            current_unique = len(self._deduplicate_results(results))
            new_unique_this_page = current_unique - previous_unique_count
            page_duplicate_rate = 1 - (new_unique_this_page / max(len(page_results), 1)) if page_results else 0
            
            logger.info(f"Cluster {cluster_num}, Page {page}: "
                       f"{len(page_results)} raw results, "
                       f"{new_unique_this_page} new unique results, "
                       f"{page_duplicate_rate:.1%} page duplicate rate")
            
            # Location-aware smart pagination logic
            should_stop = False
            
            if page >= min_pages:  # Only consider stopping after minimum pages
                if new_unique_this_page < stop_threshold:
                    stopping_reason = f"<{stop_threshold} new unique results ({new_unique_this_page})"
                    should_stop = True
                elif page_duplicate_rate > duplicate_threshold:
                    stopping_reason = f">{duplicate_threshold:.0%} duplicate rate ({page_duplicate_rate:.1%})"
                    should_stop = True
            
            if should_stop:
                logger.info(f"🛑 Stopping pagination for cluster {cluster_num} after page {page}: {stopping_reason}")
                break
            elif page >= min_pages and page < max_pages:
                logger.info(f"✅ Continuing pagination for cluster {cluster_num}: above thresholds")
            
            previous_unique_count = current_unique
        
        # Calculate cluster metrics
        unique_results = self._deduplicate_results(results)
        unique_count = len(unique_results)
        duplicate_rate = 1 - (unique_count / max(len(results), 1))
        cost_per_result = (credits_used * 0.001) / max(unique_count, 1)
        
        # Log completion summary
        if stopping_reason:
            logger.info(f"📊 Cluster {cluster_num} completed: {pages_searched}/{max_pages} pages, "
                       f"{unique_count} unique results, stopped due to {stopping_reason}")
        else:
            logger.info(f"📊 Cluster {cluster_num} completed: {pages_searched}/{max_pages} pages, "
                       f"{unique_count} unique results, reached max pages")
        
        return {
            'results': results,
            'credits_used': credits_used,
            'pages_searched': pages_searched,
            'duplicate_rate': duplicate_rate,
            'cost_per_result': cost_per_result,
            'efficiency': unique_count / max(credits_used, 1),  # Results per credit
            'stopping_reason': stopping_reason or f"reached max pages ({max_pages})",
            'pagination_strategy_used': pagination_strategy,
            'vertical_results': vertical_results  # Include results by vertical for contribution tracking
        }
    
    def _create_cluster_batch_payload(self, verticals: List[str], cluster: LocationCluster, 
                                    page: int) -> Tuple[List[Dict], List[Dict], List[int]]:
        """
        Create batch request payload for a location cluster, checking cache first.
        
        Returns:
            Tuple of (batch_payload, cached_results, cache_mapping)
        """
        batch_payload = []
        cached_results = []
        cache_mapping = []  # Maps batch index to vertical index
        
        for idx, vertical in enumerate(verticals):
            # Check cache first
            cached_data = self.search_cache.get(
                vertical=vertical,
                location=cluster.centroid_name,
                lat=cluster.centroid_lat,
                lon=cluster.centroid_lon,
                zoom=cluster.recommended_zoom,
                page=page
            )
            
            if cached_data is not None:
                # Use cached result
                cached_results.append(cached_data)
                cache_mapping.append(idx)
            else:
                # Add to batch for API call
                query = {
                    "q": f"{vertical} in {cluster.centroid_name}",
                    "ll": f"{cluster.centroid_lat},{cluster.centroid_lon},{cluster.recommended_zoom}z",
                    "page": page
                }
                batch_payload.append(query)
        
        return batch_payload, cached_results, cache_mapping
    
    def _calculate_optimization_savings(self, original_locations: int, 
                                      optimized_clusters: int, 
                                      credits_used: int, 
                                      verticals_count: int = 1) -> Dict:
        """Calculate savings from geographic optimization."""
        # Estimate what the cost would have been without optimization
        estimated_original_credits = original_locations * verticals_count * 3  # Assume 3 pages avg
        
        if estimated_original_credits > 0:
            credits_saved = max(0, estimated_original_credits - credits_used)
            savings_percentage = (credits_saved / estimated_original_credits) * 100
        else:
            credits_saved = 0
            savings_percentage = 0
        
        return {
            'estimated_original_credits': estimated_original_credits,
            'actual_credits_used': credits_used,
            'credits_saved': credits_saved,
            'savings_percentage': savings_percentage,
            'location_reduction': original_locations - optimized_clusters
        }
    
    def _execute_batch_request(self, batch_payload: List[Dict]) -> Optional[List[Dict]]:
        """Execute a batch request to Serper API using persistent session."""
        try:
            # Use the session for request (connection pooling enabled)
            response = self.session.post(
                f"https://{self.base_url}{self.endpoint}",
                json=batch_payload,
                timeout=(5, 30)  # (connect timeout, read timeout)
            )
            
            if response.status_code != 200:
                logger.error(f"Serper API error: {response.status_code} - {response.text}")
                return None
            
            batch_data = response.json()
            
            if not isinstance(batch_data, list):
                logger.error(f"Unexpected response format: {type(batch_data)}")
                return None
            
            return batch_data
            
        except requests.exceptions.Timeout:
            logger.error("Request timeout in batch request")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Connection error in batch request")
            return None
        except Exception as e:
            logger.error(f"Error in batch request: {str(e)}")
            return None
    
    def _matches_vertical(self, place: Dict, vertical: str) -> bool:
        """Check if a place matches the searched vertical using AI validation."""
        # Extract business data from place
        business_types = place.get('types', [])  # This is the key fix - using 'types' not 'categories'
        business_title = place.get('title', '')
        
        # Use AI validation if available
        if self.ai_validator:
            try:
                validation_result = self.ai_validator.validate_match(
                    vertical=vertical,
                    business_types=business_types,
                    business_title=business_title
                )
                
                # Log validation decision for debugging
                logger.debug(
                    f"AI validation: '{vertical}' vs {business_types} [{business_title}] -> "
                    f"{validation_result.is_match} ({validation_result.reasoning}, "
                    f"{validation_result.response_time_ms}ms, cache_hit={validation_result.cache_hit})"
                )
                
                return validation_result.is_match
                
            except Exception as e:
                logger.error(f"AI validation failed, using fallback: {str(e)}")
                # Fall through to fallback logic
        
        # Fallback to improved rule-based validation (using 'types' instead of 'categories')
        return self._fallback_vertical_validation(place, vertical)
    
    def _fallback_vertical_validation(self, place: Dict, vertical: str) -> bool:
        """Fallback rule-based validation using correct 'types' field."""
        place_type = place.get('type', '').lower()
        business_types = [t.lower() for t in place.get('types', [])]  # Fixed: using 'types'
        place_title = place.get('title', '').lower()
        
        vertical_lower = vertical.lower()
        
        # Check business types (main improvement - checking the right field)
        for business_type in business_types:
            if vertical_lower in business_type:
                return True
            if any(word in business_type for word in vertical_lower.split()):
                return True
        
        # Check other fields as secondary validation
        if vertical_lower in place_type:
            return True
        if vertical_lower in place_title:
            return True
        if any(word in place_type for word in vertical_lower.split()):
            return True
        if any(word in place_title for word in vertical_lower.split()):
            return True
        
        return False
    
    def _process_place(self, place: Dict, vertical: str) -> Optional[Dict]:
        """Process raw place data into standardized format with enhanced address parsing and AI classification."""
        try:
            website = place.get('website', '')
            social_flag = bool(website and (
                'facebook' in website.lower() or 
                'instagram' in website.lower()
            ))
            address = place.get('address', '')
            
            # Enhanced address parsing using new utility
            parsed_address = us_address_parser.parse_address(address)
            street_address = parsed_address.get('street_address')
            city = parsed_address.get('city') or self._extract_city(address)  # Fallback to old method
            state_code = parsed_address.get('state') or self._extract_state_code(address)  # Fallback to old method
            zip_code = parsed_address.get('zip_code')
            
            # Convert coordinates
            try:
                latitude = float(place.get('latitude', 0))
                longitude = float(place.get('longitude', 0))
            except (ValueError, TypeError):
                latitude = longitude = 0.0
            
            # AI-powered population type classification
            population_type = None
            if self.city_classifier and city:
                try:
                    classification_result = self.city_classifier.classify_city(
                        city_name=city,
                        state=state_code or '',
                        country='US'
                    )
                    population_type = classification_result.location_density.value
                    logger.debug(f"Classified {city}, {state_code} as {population_type}")
                except Exception as e:
                    logger.warning(f"Failed to classify city {city}: {str(e)}")
            
            # Extract price level from Serper data
            price_level = place.get('priceLevel') or place.get('price_level')
            
            return {
                'name': place.get('title', ''),
                'placeId': place.get('placeId', ''),
                'cid': place.get('cid', ''),
                'fid': place.get('fid', ''),
                'rating': float(place.get('rating', 0)),
                'reviews': int(place.get('ratingCount', 0)),
                'type': place.get('type', ''),
                'address': address,
                'street_address': street_address,
                'city': city,
                'stateCode': state_code,
                'zip_code': zip_code,
                'latitude': latitude,
                'longitude': longitude,
                'population_type': population_type,
                'price_level': price_level,
                'website': website,
                'phoneNumber': place.get('phoneNumber', ''),
                'openingHours': place.get('openingHours', {}),
                'thumbnailUrl': place.get('thumbnailUrl', ''),
                'vertical': vertical,
                'socialFlag': social_flag,
                'mapsUrl': self._generate_maps_url(place.get('placeId', ''))
            }
        except Exception as e:
            logger.error(f"Error processing place data: {str(e)}")
            return None
    
    def _extract_city(self, address: str) -> str:
        """Extract city from address."""
        try:
            parts = address.split(',')
            if len(parts) >= 2:
                return parts[1].strip()
            return ""
        except Exception:
            return ""
    
    def _extract_state_code(self, address: str) -> str:
        """Extract state code from address."""
        try:
            parts = address.split(", ")
            if len(parts) >= 2:
                last_part = parts[-1]
                state_code = re.findall(r'\b[A-Z]{2}\b', last_part)
                return state_code[0] if state_code else ""
            return ""
        except Exception:
            return ""
    
    def _generate_maps_url(self, place_id: str) -> str:
        """Generate Google Maps URL for a place."""
        return f"https://www.google.com/maps/place/?q=place_id:{place_id}"
    
    def _deduplicate_results(self, results: List[Dict]) -> Dict[str, Dict]:
        """Deduplicate results by name and website."""
        unique_results = {}
        website_map = {}
        
        def normalize_website(website: str) -> str:
            """Normalize website URL for comparison."""
            if not website:
                return ""
            # Remove protocol
            website = re.sub(r'^https?://', '', website.lower())
            # Remove trailing slash
            website = website.rstrip('/')
            # Remove 'www.'
            website = re.sub(r'^www\.', '', website)
            return website
        
        def is_better_entry(new_place: Dict, existing_place: Dict) -> bool:
            """Determine if the new place entry is better than existing."""
            if new_place['rating'] > existing_place['rating']:
                return True
            if new_place['rating'] == existing_place['rating']:
                if new_place['reviews'] > existing_place['reviews']:
                    return True
            return False
        
        # Process each result
        for place in results:
            name = place['name'].lower().strip()
            website = normalize_website(place.get('website', ''))
            
            # Check name-based entries
            if name in unique_results:
                if is_better_entry(place, unique_results[name]):
                    unique_results[name] = place
                    if website:
                        website_map[website] = place
            else:
                unique_results[name] = place
                if website:
                    if website in website_map:
                        # Website match with different name
                        if is_better_entry(place, website_map[website]):
                            # Remove old entry
                            old_name = website_map[website]['name'].lower().strip()
                            if old_name in unique_results:
                                del unique_results[old_name]
                            website_map[website] = place
                            unique_results[name] = place
                    else:
                        website_map[website] = place
        
        logger.info(f"Deduplicated {len(results)} results to {len(unique_results)} unique businesses")
        return unique_results
    
    def fetch_reviews(self, place_data: Dict, reviews_sort_by: str = 'mostRelevant', 
                     max_reviews: int = 50) -> List[Dict]:
        """
        Fetch reviews for a specific place using Serper Reviews API with pagination support.
        
        Args:
            place_data: Dictionary containing placeId, cid, and/or fid
            reviews_sort_by: Sort order for reviews (mostRelevant, newest, highestRating, lowestRating)
            max_reviews: Maximum number of reviews to fetch
            
        Returns:
            List of review dictionaries
        """
        try:
            url = "https://google.serper.dev/reviews"
            all_reviews = []
            total_credits = 0
            page = 1
            next_page_token = None
            
            # Build base payload - use whatever identifiers are available
            base_payload = {}
            if place_data.get('placeId'):
                base_payload['placeId'] = place_data['placeId']
            if place_data.get('cid'):
                base_payload['cid'] = place_data['cid']
            if place_data.get('fid'):
                base_payload['fid'] = place_data['fid']
                
            if not base_payload:
                logger.warning(f"No valid identifiers for fetching reviews: {place_data.get('name', 'Unknown')}")
                return []
            
            # Add sorting
            base_payload['sort'] = reviews_sort_by
            base_payload['num'] = 50  # Request max per page (though API only returns 8)
            
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            # Fetch reviews with pagination
            while len(all_reviews) < max_reviews:
                payload = base_payload.copy()
                
                # Add pagination token if available
                if next_page_token:
                    payload['pageToken'] = next_page_token
                
                # Use session for reviews fetching as well
                response = self.session.post(url, json=payload, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                page_reviews = data.get('reviews', [])
                
                if not page_reviews:
                    break
                
                all_reviews.extend(page_reviews)
                total_credits += data.get('credits', 1)
                
                # Check for next page token
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
                
                # Log progress
                logger.debug(f"Fetched page {page}: {len(page_reviews)} reviews, total: {len(all_reviews)}")
                page += 1
                
                # Limit pages to prevent excessive API calls
                if page > 10:  # Max 10 pages = ~80 reviews
                    logger.info(f"Reached page limit for {place_data.get('name', 'Unknown')}")
                    break
            
            # Trim to max_reviews if we fetched more
            if len(all_reviews) > max_reviews:
                all_reviews = all_reviews[:max_reviews]
            
            # Log the credit usage
            logger.info(f"Fetched {len(all_reviews)} reviews for {place_data.get('name', 'Unknown')} "
                       f"across {page} pages (credits: {total_credits})")
            return all_reviews
            
        except Exception as e:
            logger.error(f"Error fetching reviews for {place_data.get('name', 'Unknown')}: {str(e)}")
            return []
    
    def close(self):
        """Explicitly close the session and release resources."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup on context manager exit."""
        self.close()
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.close()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.search_cache.get_stats() if hasattr(self, 'search_cache') else {}