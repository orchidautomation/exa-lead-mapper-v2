#!/usr/bin/env python3
"""
Test script for Business Mapper API v2.0
"""
import requests
import json
import sys

# Configuration
API_URL = "http://localhost:8080"
ADMIN_KEY = "local-admin-key-456"  # From .env

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def create_test_api_key():
    """Create a test API key."""
    print("\nCreating test API key...")
    headers = {
        "X-Admin-Key": ADMIN_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "description": "Test API key for validation"
    }
    
    response = requests.post(
        f"{API_URL}/admin/api-keys",
        headers=headers,
        json=data
    )
    
    if response.status_code == 201:
        result = response.json()
        print(f"Created API key: {result['api_key']}")
        return result['api_key']
    else:
        print(f"Failed to create API key: {response.text}")
        return None

def test_search(api_key):
    """Test search endpoint."""
    print("\nTesting search endpoint...")
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Simple search request
    data = {
        "verticals": ["coffee shops"],
        "locations": ["Knoxville, TN"],
        "zoom": 14,
        "max_pages": 1,
        "output_format": "json"
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/search",
        headers=headers,
        json=data
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['unique_businesses_found']} businesses")
        print(f"Credits used: {result['total_credits_used']}")
        print(f"Cost: ${result['cost_of_credits']:.3f}")
        return True
    else:
        print(f"Search failed: {response.text}")
        return False

def main():
    """Run all tests."""
    print("🧪 Business Mapper API v2.0 Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("❌ Health check failed. Is the server running?")
        sys.exit(1)
    
    # Test 2: Create API key
    api_key = create_test_api_key()
    if not api_key:
        print("❌ Failed to create API key")
        sys.exit(1)
    
    # Test 3: Search
    if test_search(api_key):
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Search test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()