# Business Mapper API v2.0

A powerful, AI-enhanced business search API that provides intelligent location and vertical expansion for comprehensive search results.

## 🚀 Features

- **AI-Powered Search Expansion**: Automatically expands search terms and locations for better coverage
- **Location Intelligence**: Converts counties/regions into searchable cities
- **Review Analysis**: AI-powered review insights using Groq
- **Smart Caching**: Database caching for performance
- **Cost Optimization**: Predictive cost analysis and safety thresholds
- **Multiple Output Formats**: JSON and CSV support

## 📋 Prerequisites

- Python 3.11+
- Serper API key (for business search)
- Groq API key (for AI-powered query expansion)

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd mapper-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: SERPER_API_KEY, GROQ_API_KEY
# Optional: ADMIN_KEY (for admin endpoints)
```

### 3. Run the API

```bash
python app.py
```

The API will start on http://localhost:8080

## 🔑 API Authentication

### Create an API Key

First, create an API key using the admin endpoint:

```bash
curl -X POST http://localhost:8080/admin/api-keys \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-admin-key" \
  -d '{"description": "My API Key"}'
```

### Use the API Key

Include the API key in all search requests:

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "verticals": ["golf courses"],
    "locations": ["Knox County, TN"],
    "auto_expand": true
  }'
```

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Search Businesses
```
POST /api/v1/search
```

**Request Body:**
```json
{
  "verticals": ["golf courses", "country clubs"],
  "locations": ["Knox County, TN"],
  "auto_expand": true,
  "expansion_mode": "comprehensive",
  "search_intent": "Facilities with large grass areas for lawn maintenance",
  "include_reviews": true,
  "analyze_reviews": true,
  "output_format": "json"
}
```

**Key Parameters:**
- `verticals`: Business types to search for
- `locations`: Places to search (cities, counties, ZIP codes)
- `auto_expand`: Enable AI-powered vertical expansion
- `search_intent`: Context for AI to understand your needs
- `include_reviews`: Fetch review data
- `analyze_reviews`: AI analysis of reviews (requires Groq)

## 🧠 AI Features

### Location Expansion
Automatically expands counties and regions into searchable cities:
- "Knox County, TN" → Knoxville, Farragut, Powell, Karns, Halls

### Vertical Expansion
Intelligently expands search terms based on context:
- "golf courses" → golf courses, country clubs, golf clubs, golf resorts

### Search Intent
Provides context to filter and prioritize results:
- "facilities with real grass for robotic mowing" excludes mini golf, TopGolf

## 🚀 Deployment

### Railway

1. Install Railway CLI
2. Initialize project:
```bash
railway login
railway init
```

3. Deploy:
```bash
railway up
```

4. Set environment variables in Railway dashboard

### Docker

```bash
# Build image
docker build -t business-mapper .

# Run container
docker run -p 8080:8080 --env-file .env business-mapper
```

## 📊 Example: Lead Generation

Find all golf facilities in Knox County for a robotic mowing company:

```python
import requests

api_key = "your-api-key"
url = "http://localhost:8080/api/v1/search"

payload = {
    "verticals": [
        "golf course",
        "country club",
        "golf resort"
    ],
    "locations": ["Knox County, TN"],
    "auto_expand": True,
    "expansion_mode": "comprehensive",
    "search_intent": "Golf facilities with extensive grass areas suitable for robotic lawn mowing services",
    "include_reviews": True,
    "analyze_reviews": True,
    "output_format": "csv"
}

headers = {
    "X-API-Key": api_key,
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

# Save CSV
with open("golf_facilities.csv", "wb") as f:
    f.write(response.content)
```

## 🔧 Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SERPER_API_KEY` | Serper API key for search | Required |
| `GROQ_API_KEY` | Groq API key for AI features | Required |
| `ADMIN_KEY` | Admin key for management | Optional |
| `DEFAULT_MAX_PAGES` | Max search result pages | 3 |
| `SEARCH_COST_SAFETY_THRESHOLD_USD` | Cost limit per search | 1.0 |

## 📈 Performance

- **Caching**: Location geocoding cached for 24 hours
- **Concurrent Requests**: Up to 5 parallel searches
- **Smart Pagination**: Stops when unique results decrease
- **Cost Control**: Automatic safety thresholds

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.