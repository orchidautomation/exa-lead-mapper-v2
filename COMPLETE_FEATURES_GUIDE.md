# Mapper V2 Complete Feature Documentation

## 🚀 Core Capabilities

### 1. **AI-Powered Location Expansion**
```python
# When you provide a county, region, or area:
"locations": ["Knox County, TN"]

# V2 automatically expands to:
- All cities within the county
- Unincorporated areas
- Suburbs and communities
- Neighboring relevant areas
```

**Example**: Knox County → Knoxville, Farragut, Powell, Halls, Karns, Corryton, Mascot, Gibbs, Heiskell, Strawberry Plains, Cedar Bluff, Fountain City, etc.

### 2. **AI-Powered Vertical Expansion**
```python
# Input: General business type
"verticals": ["golf course"]

# V2 intelligently expands based on search_intent to:
- Related business types
- Variations and synonyms
- Industry-specific terms
```

## 📊 Complete Parameter Reference

### **Required Parameters**
```json
{
  "verticals": ["business type 1", "business type 2"],  // At least 1 required
  "locations": ["location 1", "location 2"]             // At least 1 required
}
```

### **All Optional Parameters**
```json
{
  // AI Expansion Controls
  "auto_expand": true,                    // Enable AI-powered expansion (default: false)
  "expansion_mode": "comprehensive",      // "minimal" | "balanced" | "comprehensive"
  "expansion_priority": "coverage",       // "cost_control" | "balanced" | "coverage"
  
  // Search Intent - CRITICAL for AI understanding
  "search_intent": "Detailed description of what you're looking for and why",
  
  // Search Configuration
  "zoom": 13,                            // Map zoom level (1-21, default: 14)
  "max_pages": 8,                        // Max result pages per location (1-8)
  
  // Cost Controls
  "max_credit_budget": 1000,             // Maximum credits to spend
  "max_cost_multiplier": 3.0,            // Max cost increase from expansion (1.0-10.0)
  
  // Review Analysis
  "include_reviews": true,               // Fetch review data
  "analyze_reviews": true,               // AI analysis of reviews
  "reviews_sort_by": "newest",           // "highestRating" | "mostRelevant" | "newest" | "lowestRating"
  "max_reviews_analyze": 50,             // Reviews to analyze per business (20-100)
  "min_reviews_for_analysis": 10,        // Minimum reviews required for analysis (5-50)
  
  // Output Format
  "output_format": "csv"                 // "json" | "csv"
}
```

## 🎯 Search Intent Examples

The `search_intent` parameter is **CRITICAL** for proper AI expansion:

```json
{
  "search_intent": "Golf facilities with extensive natural grass areas suitable for robotic lawn mowing services. Focus on full-size golf courses and country clubs with large maintained grass areas. Exclude mini golf, TopGolf, driving ranges only, and indoor facilities."
}
```

This tells the AI to:
- Focus on facilities with large grass areas
- Exclude certain types (mini golf, indoor)
- Understand the business purpose (robotic mowing services)

## 🌍 Location Expansion Features

### **Types Detected**:
- **County**: "Knox County" → All cities and communities
- **Region**: "East Tennessee" → Multiple counties
- **Metro Area**: "Knoxville Metro" → City + suburbs
- **Neighborhood**: "West Knoxville" → Specific area

### **Expansion Intelligence**:
```python
# AI considers:
- Population density
- Business density
- Geographic boundaries
- Administrative divisions
- Unincorporated areas
```

## 📈 Output Fields (CSV Format)

```csv
# Business Information
name                    # Business name
placeId                 # Google Place ID
cid                     # Google CID
type                    # Business type
vertical                # Search vertical used

# Location Data
address                 # Full address
street_address          # Street portion
city                    # City name
stateCode              # State code
zip_code               # ZIP code
latitude               # Coordinates
longitude              # Coordinates
population_type        # urban/suburban/rural/dense_urban

# Business Details
rating                 # Average rating (0-5)
reviews               # Review count
price_level           # Price indicator
website               # Website URL
phoneNumber           # Phone number
thumbnailUrl          # Image URL
mapsUrl              # Google Maps link

# Review Analysis (if enabled)
has_reviews_data      # Yes/No
reviews_analyzed      # Count analyzed
average_rating        # Calculated average
maintenance_rating    # Category rating
facilities_rating     # Category rating
course_conditions_rating # Category rating
top_insights         # Key findings
maintenance_opportunities # Opportunities identified
```

## 🔧 Advanced Features

### **1. Intelligent Pagination**
- Stops automatically when duplicate rate is high
- Adjusts based on city type (urban/suburban/rural)
- Cost-aware page limiting

### **2. Geographic Optimization**
- Clusters nearby locations
- Adjusts zoom levels by area type
- Prevents redundant searches

### **3. Term Contribution Analysis**
```json
{
  "term_contribution_analysis": {
    "term": "golf course",
    "unique_results": 45,
    "contribution_rate": 0.42,
    "cost_efficiency": 0.89
  }
}
```

### **4. Cost Optimization**
- Real-time cost tracking
- Predictive cost analysis
- Automatic search optimization

### **5. Validation & Quality**
- AI-powered result validation
- Business type verification
- Location accuracy checking

## 🎛️ Expansion Modes Explained

### **minimal**
- Conservative expansion
- Adds 2-3 most relevant terms/locations
- Lowest cost increase

### **balanced** (default)
- Moderate expansion
- Adds 4-6 relevant terms/locations
- Good cost/coverage balance

### **comprehensive**
- Maximum expansion
- Adds all relevant terms/locations
- Highest coverage, higher cost

## 📊 Response Metadata

```json
{
  "search_metadata": {
    "session_id": "unique-id",
    "search_intent": "your-intent",
    "location_expansion": {
      "expansion_performed": true,
      "original_locations": ["Knox County, TN"],
      "expanded_locations": ["Knoxville", "Farragut", ...],
      "expansion_details": [
        {
          "original": "Knox County, TN",
          "expanded_to": ["cities..."],
          "type": "county",
          "reasoning": "AI reasoning"
        }
      ]
    },
    "vertical_expansion": {
      "original_terms": "golf course",
      "expanded_terms": ["golf course", "country club", ...],
      "ai_enabled": true,
      "expansion_reasoning": "AI reasoning"
    },
    "total_credits_used": 84,
    "total_cost": 0.084,
    "unique_results_count": 106,
    "total_results": 250
  }
}
```

## 💡 Pro Tips for Maximum Results

1. **Always include search_intent** - This dramatically improves AI expansion accuracy
2. **Use comprehensive mode** for exhaustive searches
3. **Set high max_pages** (8) for thorough coverage
4. **Include broad geographic areas** (counties, regions) and let AI expand
5. **Enable review analysis** for business insights

## 🚨 Limits & Constraints

- Max 8 pages per location cluster
- Max 100 reviews analyzed per business
- Locations must be geocodable
- Cost safety thresholds may block very large searches
- API key required for authentication

## 📝 Example: Exhaustive Golf Course Search

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "verticals": ["golf course", "country club", "disc golf"],
    "locations": ["Knox County, TN"],
    "auto_expand": true,
    "expansion_mode": "comprehensive",
    "expansion_priority": "coverage",
    "search_intent": "ALL golf facilities in Knox County including country clubs, public courses, private courses, municipal courses, disc golf, and any facility with golf in the name. Include everything - we want exhaustive coverage.",
    "include_reviews": false,
    "analyze_reviews": false,
    "zoom": 11,
    "max_pages": 8,
    "output_format": "csv",
    "max_cost_multiplier": 10.0,
    "max_credit_budget": 1000
  }'
```

This search found **106 unique golf facilities** across Knox County by:
- Expanding to 14+ cities and communities
- Using AI to optimize search terms
- Intelligent pagination to maximize results while minimizing cost
- Total cost: $0.084 (84 credits)