#!/usr/bin/env python3
"""Test script to verify batch AI validation is working properly."""

import requests
import json
import time

API_KEY = "Wqc4u9hY3Sp69rd4CnWwQm4c4KBd-Q8ad8LOn-puAKg"
BASE_URL = "http://127.0.0.1:8080/api/v1"

def test_search_with_batch_validation():
    """Test search endpoint with multiple verticals to trigger batch validation."""
    
    # Search with multiple verticals to ensure batch processing
    payload = {
        "verticals": ["coffee shop", "restaurant", "cafe", "bakery", "bar", "brewery"],
        "locations": ["Austin, TX"],
        "max_pages": 2,
        "include_reviews": False,
        "analyze_content": False
    }
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("🚀 Testing batch AI validation with 6 verticals...")
    print(f"Verticals: {payload['verticals']}")
    print(f"Location: {payload['locations'][0]}")
    print("-" * 50)
    
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/search",
        headers=headers,
        json=payload
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"✅ Search completed in {elapsed_time:.2f} seconds")
        print(f"Total results: {data.get('unique_businesses_found', 0)}")
        print(f"Total credits used: {data.get('total_credits_used', 0)}")
        print(f"Cost: ${data.get('cost_of_credits', 0):.4f}")
        
        # Check search metadata
        metadata = data.get('search_metadata', {})
        print("\n📊 Search Metadata:")
        print(f"  - Verticals searched: {metadata.get('verticals_searched', 0)}")
        print(f"  - Original locations: {metadata.get('original_locations', 0)}")
        print(f"  - Optimized clusters: {metadata.get('optimized_clusters', 0)}")
        
        # Check cache stats
        cache_stats = metadata.get('cache_stats', {})
        if cache_stats:
            print("\n💾 Cache Statistics:")
            print(f"  - Cache size: {cache_stats.get('size', 0)}")
            print(f"  - Hits: {cache_stats.get('hits', 0)}")
            print(f"  - Misses: {cache_stats.get('misses', 0)}")
            print(f"  - Hit rate: {cache_stats.get('hit_rate', 0)}%")
        
        # Save response for analysis
        with open('batch_validation_test_results.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Full results saved to batch_validation_test_results.json")
        
        # Look for batch validation indicators in logs
        print("\n🔍 Checking server logs for batch validation...")
        try:
            with open('server.log', 'r') as f:
                log_content = f.read()
                if "Batch AI validation" in log_content:
                    print("✅ Batch AI validation detected in logs!")
                if "Batch validation completed" in log_content:
                    print("✅ Batch validation completed successfully!")
        except:
            print("⚠️ Could not read server logs")
        
        return data
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

def check_validation_stats():
    """Check AI validation statistics."""
    headers = {"X-API-Key": API_KEY}
    
    # This endpoint doesn't exist yet, but would be useful
    # For now, we'll check the logs
    print("\n📊 Checking validation statistics...")
    
    try:
        with open('server.log', 'r') as f:
            content = f.read()
            
            # Count batch validation occurrences
            batch_count = content.count("Batch validation completed")
            cache_hits = content.count("Cache hit for key")
            
            print(f"  - Batch validations: {batch_count}")
            print(f"  - Cache hits: {cache_hits}")
    except:
        print("⚠️ Could not read server logs")

if __name__ == "__main__":
    print("🧪 Batch AI Validation Test Suite")
    print("=" * 50)
    
    # Run the test
    result = test_search_with_batch_validation()
    
    # Check stats
    check_validation_stats()
    
    print("\n✅ Test completed!")