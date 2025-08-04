# Business Mapper API v2.0 - Quick Start Guide

## 🚀 Quick Setup

1. **Clone and setup**:
   ```bash
   cd mapper-v2
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   ./start.sh
   # Or directly: python app.py
   ```

## 🔑 Required API Keys

Get these API keys and add them to your `.env` file:

1. **Serper API Key** (required for business search)
   - Sign up at: https://serper.dev
   - Add to `.env`: `SERPER_API_KEY=your_key_here`

2. **Groq API Key** (required for AI features)
   - Sign up at: https://groq.com
   - Add to `.env`: `GROQ_API_KEY=your_key_here`

3. **Admin Key** (for API key management)
   - Set a secure value: `ADMIN_KEY=your_secure_admin_key`

## 📡 API Endpoints

### 1. Health Check
```bash
curl http://localhost:8080/api/v1/health
```

### 2. Create API Key (Admin Only)
```bash
curl -X POST http://localhost:8080/api/v1/admin/create-api-key \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_admin_key" \
  -d '{"description": "My test key"}'
```

### 3. Basic Business Search
```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "verticals": ["coffee shop"],
    "locations": ["Austin, TX"],
    "output_format": "json"
  }'
```

### 4. Advanced Search with AI Features
```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "verticals": ["outdoor recreation"],
    "locations": ["Knox County, TN"],
    "auto_expand": true,
    "expansion_mode": "comprehensive",
    "search_intent": "facilities with real grass for lawn maintenance",
    "include_reviews": true,
    "analyze_reviews": true,
    "output_format": "csv"
  }'
```

## 🔧 Search Parameters

### Required
- `verticals`: Array of business types (e.g., ["coffee shop", "restaurant"])
- `locations`: Array of locations (e.g., ["Austin, TX", "Portland, OR"])

### Optional
- `zoom`: Map zoom level (1-21, default: 14)
- `max_pages`: Maximum result pages (1-8, auto-determined if not set)
- `output_format`: "json" or "csv" (default: "json")

### AI Expansion
- `auto_expand`: Enable AI vertical expansion (default: false)
- `expansion_mode`: "minimal", "balanced", or "comprehensive"
- `expansion_priority`: "cost_control", "balanced", or "coverage"
- `search_intent`: Context for AI expansion (e.g., "pet-friendly locations")

### Reviews Analysis
- `include_reviews`: Include raw review data (default: false)
- `analyze_reviews`: Enable AI review analysis (default: false)
- `reviews_sort_by`: "mostRelevant", "newest", "highestRating", "lowestRating"
- `max_reviews_analyze`: Max reviews to analyze per business (20-100)

## 📊 Response Examples

### JSON Response
```json
{
  "results": [
    {
      "name": "Austin Coffee Roasters",
      "rating": 4.5,
      "reviews": 1247,
      "type": "Coffee shop",
      "address": "123 South Lamar Blvd, Austin, TX 78704",
      "city": "Austin",
      "stateCode": "TX",
      "latitude": 30.2672,
      "longitude": -97.7431,
      "population_type": "urban",
      "website": "https://austincoffee.com",
      "phoneNumber": "+1-512-555-0123"
    }
  ],
  "total_credits_used": 5,
  "cost_of_credits": 0.005,
  "unique_businesses_found": 23,
  "search_metadata": {
    "search_timestamp": "2024-01-01T12:00:00Z",
    "locations_searched": 1,
    "verticals_searched": 1
  }
}
```

### CSV Output
When `output_format: "csv"`, you'll get a downloadable CSV file with columns:
- name, type, rating, reviews, address, city, state
- latitude, longitude, phone, website, maps_url
- population_type, price_level

## 🧠 AI Features

### 1. Vertical Expansion
Transforms broad categories into specific terms:
- Input: "outdoor recreation"
- Output: ["golf courses", "disc golf", "tennis courts", "hiking trails"]

### 2. Location Expansion  
Expands counties/regions into specific cities:
- Input: "Knox County, TN"
- Output: ["Knoxville, TN", "Farragut, TN", "Powell, TN"]

### 3. Review Analysis
AI-powered insights from business reviews:
- Top praises and problems
- Confidence scores and supporting quotes
- Rating distribution analysis

## 🐳 Docker Deployment

### Local Development
```bash
docker-compose up --build
```

### Production
```bash
docker build -t mapper-api .
docker run -p 8080:8080 --env-file .env mapper-api
```

## 🚀 Railway Deployment

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard:
   - `SERPER_API_KEY`
   - `GROQ_API_KEY`
   - `ADMIN_KEY`
   - `SECRET_KEY`
   - `DATABASE_URL` (Railway will provide PostgreSQL)
3. Deploy automatically

## 🔍 Testing

Run the setup test:
```bash
python test_setup.py
```

Test the API manually:
```bash
# Health check
curl http://localhost:8080/api/v1/health

# Create API key
curl -X POST http://localhost:8080/api/v1/admin/create-api-key \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_admin_key" \
  -d '{"description": "Test key"}'

# Test search
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"verticals": ["coffee"], "locations": ["Austin, TX"]}'
```

## ❗ Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the mapper-v2 directory
2. **API key errors**: Check your .env file has the required keys
3. **Database errors**: Ensure the data/ directory exists and is writable
4. **Geocoding failures**: Check internet connection and location format
5. **Search failures**: Verify Serper API key is valid

### Logs
Check the console output for detailed error messages and debugging information.

### Health Check
Always start with checking the health endpoint to verify all services are working.