"""
Business Mapper API v2.0 - Simplified
A clean API for business search with AI-powered expansion
"""
import os
import logging
import uuid
import csv
import io
from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
from datetime import datetime

from config.settings import settings
from services.database import DatabaseManager
from services.geocoding import OpenMeteoGeocoding
from services.serper import SerperMapsService
from services.search_optimizer import SearchOptimizer
from services.vertical_expander import CostAwareExpander
from services.location_expander import LocationExpander
from services.reviews_analyzer import ReviewsAnalyzer
from utils.geographic import GeographicOptimizer
from models.schemas import SearchRequest, SearchResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Flask configuration
    app.config['SECRET_KEY'] = settings.SECRET_KEY
    app.config['JSON_SORT_KEYS'] = False
    
    # Enable CORS for all endpoints
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "X-API-Key"]
        }
    })
    
    # Initialize database
    os.makedirs('data', exist_ok=True)
    db_manager = DatabaseManager(settings.DATABASE_PATH)
    
    # Initialize services
    geocoding_service = OpenMeteoGeocoding(db_manager)
    search_optimizer = SearchOptimizer(db_manager)
    geo_optimizer = GeographicOptimizer()
    vertical_expander = CostAwareExpander(geo_optimizer, db_manager)
    location_expander = LocationExpander(geocoding_service, db_manager)
    reviews_analyzer = ReviewsAnalyzer()
    
    @app.before_request
    def before_request():
        """Set up request context."""
        g.db_manager = db_manager
        g.geocoding_service = geocoding_service
        g.search_optimizer = search_optimizer
        g.vertical_expander = vertical_expander
        g.location_expander = location_expander
        g.reviews_analyzer = reviews_analyzer
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """
        Health check endpoint.
        Returns the status of all services and dependencies.
        """
        try:
            # Check database
            db_status = "healthy"
            try:
                # Only check if database is accessible, not API key validation
                if hasattr(g, 'db_manager') and g.db_manager:
                    db_status = "healthy"
            except:
                db_status = "unavailable"
            
            # Check geocoding
            geocoding_status = "healthy"
            try:
                if hasattr(g, 'geocoding_service') and g.geocoding_service:
                    geocoding_status = "healthy"
            except:
                geocoding_status = "unavailable"
            
            # API key status
            serper_status = "configured" if settings.SERPER_API_KEY else "missing"
            groq_status = "configured" if settings.GROQ_API_KEY else "missing"
            
            # Overall status - only fail if critical services are missing
            if serper_status == "missing" or groq_status == "missing":
                overall_status = "degraded"
                status_code = 503
            else:
                overall_status = "healthy"
                status_code = 200
            
            return jsonify({
                "status": overall_status,
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "database": db_status,
                    "geocoding": geocoding_status,
                    "serper_api": serper_status,
                    "groq_api": groq_status
                },
                "environment": settings.FLASK_ENV
            }), status_code
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.route('/search', methods=['POST'])
    def search():
        """
        Main search endpoint with AI-powered expansion.
        
        Expected JSON payload:
        {
            "verticals": ["business type 1", "business type 2"],
            "locations": ["location 1", "location 2"],
            "auto_expand": true,
            "include_reviews": false,
            "output_format": "json"
        }
        """
        # API key validation removed - personal use only
        api_key_id = None  # No authentication required
        
        try:
            # Parse request
            search_request = SearchRequest(**request.json)
            session_id = str(uuid.uuid4())
            
            logger.info(f"Search {session_id}: {len(search_request.verticals)} verticals, "
                       f"{len(search_request.locations)} locations")
            
            # Expand locations if needed
            all_locations_to_search = []
            location_expansion_details = []
            
            for location in search_request.locations:
                expansion_result = g.location_expander.expand_location(
                    location=location,
                    business_verticals=search_request.verticals,
                    search_intent=search_request.search_intent,
                    max_locations=5,
                    include_suburbs=True
                )
                
                if len(expansion_result.selected_locations) > 1:
                    all_locations_to_search.extend(expansion_result.selected_locations)
                    location_expansion_details.append({
                        'original': location,
                        'expanded_to': expansion_result.selected_locations,
                        'type': expansion_result.detected_type.value,
                        'reasoning': expansion_result.expansion_reasoning
                    })
                else:
                    all_locations_to_search.append(location)
            
            # Update locations
            search_request.locations = all_locations_to_search
            
            # Expand verticals if requested
            vertical_expansion_details = None
            if search_request.auto_expand:
                expanded_verticals = []
                
                for vertical in search_request.verticals:
                    expansion_result = g.vertical_expander.expand_vertical(
                        vertical=vertical,
                        location=search_request.locations[0] if search_request.locations else None,
                        expansion_mode=search_request.expansion_mode,
                        search_intent=search_request.search_intent
                    )
                    
                    if expansion_result.selected_verticals:
                        expanded_verticals.extend(expansion_result.selected_verticals)
                    else:
                        expanded_verticals.append(vertical)
                
                # Remove duplicates
                seen = set()
                unique_expanded = []
                for v in expanded_verticals:
                    if v not in seen:
                        seen.add(v)
                        unique_expanded.append(v)
                
                vertical_expansion_details = {
                    'original_terms': search_request.verticals,
                    'expanded_terms': unique_expanded,
                    'expansion_count': len(unique_expanded) - len(search_request.verticals)
                }
                search_request.verticals = unique_expanded
            
            # Geocode locations
            geocoded_locations = []
            for location in search_request.locations:
                coords = g.geocoding_service.geocode(location)
                if coords:
                    geocoded_locations.append((location, coords[0], coords[1]))
                else:
                    logger.warning(f"Failed to geocode: {location}")
            
            if not geocoded_locations:
                return jsonify({"error": "Failed to geocode any locations"}), 400
            
            # Check search cost
            prediction = g.search_optimizer.analyze_search_request(
                search_request.verticals,
                geocoded_locations,
                search_request.zoom,
                search_request.max_pages
            )
            
            should_proceed, message = g.search_optimizer.should_proceed_with_search(prediction)
            if not should_proceed:
                return jsonify({
                    "error": "Search optimization warning",
                    "detail": message
                }), 400
            
            # Perform search
            serper_service = SerperMapsService(settings.SERPER_API_KEY, db_manager)
            search_response = serper_service.search(
                search_request.verticals,
                geocoded_locations,
                search_request.zoom,
                search_request.max_pages
            )
            
            results = search_response["results"]
            search_metadata = search_response["search_metadata"]
            search_metadata["session_id"] = session_id
            search_metadata["location_expansion"] = location_expansion_details
            search_metadata["vertical_expansion"] = vertical_expansion_details
            
            # Include reviews if requested
            if search_request.include_reviews:
                reviews_credits = 0
                for result in results:
                    if result.get('reviews', 0) >= search_request.min_reviews_for_analysis:
                        reviews_data = serper_service.fetch_reviews(
                            result,
                            reviews_sort_by=search_request.reviews_sort_by,
                            max_reviews=search_request.max_reviews_analyze
                        )
                        if reviews_data and search_request.analyze_reviews:
                            analysis = g.reviews_analyzer.analyze_reviews(
                                reviews_data[:search_request.max_reviews_analyze],
                                result['name'],
                                search_request.min_reviews_for_analysis
                            )
                            if analysis:
                                result['reviews_analysis'] = analysis.dict()
                        
                        if reviews_data:
                            pages_needed = (len(reviews_data) + 7) // 8
                            reviews_credits += pages_needed
                
                search_metadata["reviews_credits_used"] = reviews_credits
            
            # Log search
            db_manager.log_search(
                session_id,
                api_key_id,
                request.json,
                len(results),
                search_response.get('unique_businesses_found', 0),
                search_response.get('total_credits_used', 0),
                search_response.get('cost_of_credits', 0)
            )
            
            # Format response
            if search_request.output_format == 'csv':
                return _format_csv_response(results, search_metadata)
            
            return jsonify({
                "results": results,
                "metadata": search_metadata,
                "credits_used": search_response.get('total_credits_used', 0),
                "cost": search_response.get('cost_of_credits', 0)
            })
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return jsonify({"error": "Invalid request", "detail": str(e)}), 400
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route('/admin/api-keys', methods=['POST'])
    def create_api_key():
        """
        Create a new API key.
        
        JSON payload:
        {
            "description": "Description for the API key"
        }
        """
        try:
            data = request.get_json() or {}
            description = data.get('description', 'API Key')
            
            import secrets
            new_key = f"map_{secrets.token_urlsafe(32)}"
            
            if db_manager.create_api_key(new_key, description):
                return jsonify({
                    "api_key": new_key,
                    "description": description,
                    "created_at": datetime.now().isoformat()
                })
            else:
                return jsonify({"error": "Failed to create API key"}), 500
                
        except Exception as e:
            logger.error(f"API key creation error: {e}")
            return jsonify({"error": "Internal server error"}), 500
    
    @app.route('/')
    def index():
        """API information endpoint."""
        return jsonify({
            "name": "Business Mapper API",
            "version": "2.0.0",
            "description": "AI-powered business search with intelligent expansion",
            "endpoints": {
                "search": {
                    "path": "/search",
                    "method": "POST",
                    "description": "Search for businesses with location and vertical expansion"
                },
                "health": {
                    "path": "/health",
                    "method": "GET",
                    "description": "Check API and service health"
                }
            }
        })
    
    def _format_csv_response(results, metadata):
        """Format results as CSV response."""
        output = io.StringIO()
        
        if results:
            fieldnames = [
                'name', 'type', 'address', 'city', 'state', 'zip_code',
                'latitude', 'longitude', 'rating', 'reviews', 'phone',
                'website', 'maps_url', 'price_level'
            ]
            
            # Add review analysis fields if present
            if any('reviews_analysis' in r for r in results):
                fieldnames.extend([
                    'sentiment_score', 'overall_sentiment', 'top_themes'
                ])
            
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for result in results:
                row = {k: result.get(k, '') for k in fieldnames}
                
                # Add review analysis if present
                if 'reviews_analysis' in result:
                    analysis = result['reviews_analysis']
                    row['sentiment_score'] = analysis.get('sentiment_score', '')
                    row['overall_sentiment'] = analysis.get('overall_sentiment', '')
                    themes = analysis.get('key_themes', [])
                    row['top_themes'] = '; '.join(themes[:3])
                
                writer.writerow(row)
        
        # Add metadata as comments
        output.write(f"\n# Search Results Summary\n")
        output.write(f"# Total results: {len(results)}\n")
        output.write(f"# Credits used: {metadata.get('total_credits_used', 0)}\n")
        output.write(f"# Session ID: {metadata.get('session_id', '')}\n")
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename=search_results.csv'
            }
        )
    
    logger.info("Business Mapper API initialized")
    return app


if __name__ == '__main__':
    # Validate settings
    try:
        settings.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        exit(1)
    
    # Create and run app
    app = create_app()
    
    logger.info(f"Starting Business Mapper API on {settings.HOST}:{settings.PORT}")
    logger.info(f"Environment: {settings.FLASK_ENV}")
    
    app.run(
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG
    )