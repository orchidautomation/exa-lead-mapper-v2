# Business Mapper API v2.0 - Railway Deployment Guide

This guide provides step-by-step instructions for deploying the Business Mapper API v2.0 on Railway.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
3. [Environment Variables](#environment-variables)
4. [Database Configuration](#database-configuration)
5. [Deployment Steps](#deployment-steps)
6. [Post-Deployment](#post-deployment)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying to Railway, ensure you have:

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Account**: Your code should be in a GitHub repository
3. **API Keys Ready**:
   - Serper API key (for Google Maps search)
   - Groq API key (for AI analysis)
   - Admin key (you'll create this)

## Project Setup

### 1. Prepare Your Repository

Your project structure should look like this:
```
mapper-v2/
├── app.py
├── requirements.txt
├── railway.json
├── Procfile
├── api/
├── services/
├── models/
├── utils/
├── config/
└── .gitignore
```

### 2. Create Railway Configuration Files

#### railway.json
Create a `railway.json` file in your project root:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "runtime": "python",
    "startCommand": "python app.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### Procfile
Create a `Procfile` for the web process:
```
web: python app.py
```

#### runtime.txt (Optional)
Specify Python version:
```
python-3.10.12
```

### 3. Update app.py for Production

Modify your `app.py` to use Railway's PORT environment variable:

```python
import os

# At the bottom of app.py, update the run configuration:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(
        host='0.0.0.0',  # Important for Railway
        port=port,
        debug=False  # Disable debug in production
    )
```

## Environment Variables

Railway uses environment variables for configuration. You'll need to set these in the Railway dashboard:

### Required Environment Variables

```bash
# API Keys
SERPER_API_KEY=your_serper_api_key_here
GROQ_API_KEY=your_groq_api_key_here
ADMIN_KEY=generate_a_secure_admin_key

# Database (Railway will auto-populate these when you add PostgreSQL)
DATABASE_URL=postgresql://user:password@host:port/dbname

# Optional Configuration
LOG_LEVEL=INFO
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

### Generating a Secure Admin Key

Use this Python snippet to generate a secure admin key:
```python
import secrets
print(secrets.token_urlsafe(32))
```

## Database Configuration

### 1. Add PostgreSQL to Your Railway Project

1. In Railway dashboard, click "New Service"
2. Select "Database" → "Add PostgreSQL"
3. Railway will automatically set the `DATABASE_URL` environment variable

### 2. Update Database Configuration

Ensure your `config/settings.py` uses the Railway database URL:

```python
import os
from urllib.parse import urlparse

class Settings:
    # Parse DATABASE_URL for Railway
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    if DATABASE_URL:
        url = urlparse(DATABASE_URL)
        DB_CONFIG = {
            'host': url.hostname,
            'port': url.port,
            'user': url.username,
            'password': url.password,
            'database': url.path[1:]  # Remove leading '/'
        }
    else:
        # Fallback for local development
        DB_CONFIG = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'database': os.getenv('DB_NAME', 'business_mapper')
        }
```

## Deployment Steps

### 1. Connect GitHub Repository

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Authorize Railway to access your GitHub
5. Select your repository

### 2. Configure Environment Variables

1. Click on your service in Railway
2. Go to "Variables" tab
3. Add all required environment variables:
   ```
   SERPER_API_KEY=your_key
   GROQ_API_KEY=your_key
   ADMIN_KEY=your_generated_key
   ```

### 3. Add PostgreSQL Service

1. Click "New" → "Database" → "Add PostgreSQL"
2. Railway will automatically link it to your app
3. The `DATABASE_URL` will be set automatically

### 4. Deploy

1. Railway will automatically deploy when you push to your main branch
2. You can also trigger manual deploys from the dashboard
3. Monitor the build logs for any errors

### 5. Initialize Database

After first deployment, run database initialization:

1. Go to your service in Railway
2. Click on "Settings" → "Run Command"
3. Run: `python -c "from services.database import DatabaseManager; db = DatabaseManager(); db.init_db()"`

## Post-Deployment

### 1. Get Your API URL

Your API will be available at:
```
https://your-app-name.up.railway.app
```

### 2. Create Your First API Key

Use curl to create an API key:
```bash
curl -X POST https://your-app-name.up.railway.app/admin/api-keys \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: your-admin-key" \
  -d '{"description": "Production API key"}'
```

### 3. Test Your Deployment

Health check:
```bash
curl https://your-app-name.up.railway.app/api/v1/health
```

Test search:
```bash
curl -X POST https://your-app-name.up.railway.app/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "verticals": ["coffee shop"],
    "locations": ["Nashville, TN"],
    "max_pages": 1
  }'
```

## Monitoring & Maintenance

### 1. View Logs

- Go to your service in Railway
- Click on "Logs" tab
- Use filters to find specific log entries

### 2. Monitor Usage

Railway provides metrics for:
- Memory usage
- CPU usage
- Network traffic
- Response times

### 3. Set Up Alerts (Optional)

1. Go to Settings → Notifications
2. Configure alerts for:
   - Deployment failures
   - High resource usage
   - Errors

### 4. Database Backups

Railway PostgreSQL includes automatic daily backups. For manual backups:

```bash
# Connect to Railway CLI
railway login

# Select your project
railway link

# Run pg_dump
railway run pg_dump $DATABASE_URL > backup.sql
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Binding Error
**Error**: `Error: That port is already in use.`
**Solution**: Ensure your app uses `PORT` environment variable:
```python
port = int(os.environ.get('PORT', 8080))
```

#### 2. Database Connection Failed
**Error**: `psycopg2.OperationalError: could not connect to server`
**Solution**: 
- Check if PostgreSQL service is running
- Verify DATABASE_URL is set correctly
- Ensure database is initialized

#### 3. Module Import Errors
**Error**: `ModuleNotFoundError: No module named 'xxx'`
**Solution**: 
- Ensure all dependencies are in `requirements.txt`
- Check for case sensitivity in imports (Linux is case-sensitive)

#### 4. Memory Limit Exceeded
**Error**: `Container exceeded memory limit`
**Solution**:
- Optimize your code to use less memory
- Upgrade to a paid Railway plan for more resources
- Implement pagination for large datasets

#### 5. Request Timeouts
**Error**: `504 Gateway Timeout`
**Solution**:
- Railway has a 5-minute timeout limit
- For long-running searches, implement async processing
- Consider breaking large searches into smaller chunks

### Debug Commands

Run these via Railway's command runner:

```bash
# Check Python version
python --version

# List installed packages
pip list

# Test database connection
python -c "from services.database import DatabaseManager; db = DatabaseManager(); print('DB connected!')"

# Check environment variables
python -c "import os; print({k:v for k,v in os.environ.items() if 'KEY' not in k})"
```

## Best Practices

### 1. Security
- Never commit `.env` files to Git
- Use strong admin keys
- Rotate API keys regularly
- Use Railway's private networking for database connections

### 2. Performance
- Enable connection pooling for PostgreSQL
- Implement caching for frequently accessed data
- Use pagination for large result sets
- Monitor and optimize slow queries

### 3. Scaling
- Railway automatically scales your app
- For high traffic, consider:
  - Implementing rate limiting
  - Using Redis for caching
  - Horizontal scaling with multiple instances

### 4. Cost Optimization
- Monitor your usage in Railway dashboard
- Set up spending alerts
- Optimize database queries
- Use efficient data structures

## Additional Resources

- [Railway Documentation](https://docs.railway.app/)
- [Railway CLI](https://docs.railway.app/develop/cli)
- [PostgreSQL on Railway](https://docs.railway.app/databases/postgresql)
- [Environment Variables](https://docs.railway.app/develop/variables)
- [Deployment Triggers](https://docs.railway.app/deploy/deployments)

## Support

For Railway-specific issues:
- [Railway Community](https://discord.gg/railway)
- [Railway Status](https://status.railway.app/)

For Business Mapper API issues:
- Check the logs in Railway dashboard
- Review this documentation
- Ensure all environment variables are set correctly