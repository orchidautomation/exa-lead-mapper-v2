#!/usr/bin/env python3
"""
Test that individual reviews data is excluded from output
"""

import requests
import json

# API Configuration
API_KEY = "Wqc4u9hY3Sp69rd4CnWwQm4c4KBd-Q8ad8LOn-puAKg"
BASE_URL = "http://127.0.0.1:8080/api/v1"

def test_no_reviews_data():
    """Test that reviews_data is not in output when analyze_reviews is True."""
    print("🧪 Testing Reviews Data Exclusion")
    print("=" * 60)
    
    # Minimal search payload
    payload = {
        "verticals": ["golf course"],
        "locations": ["Farragut, TN"],
        "auto_expand": False,
        "max_pages": 1,
        "include_reviews": True,
        "analyze_reviews": True,
        "max_reviews_analyze": 30,
        "min_reviews_for_analysis": 10
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/search",
            json=payload,
            headers={
                "X-API-Key": API_KEY,
                "Content-Type": "application/json"
            },
            timeout=300
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            print(f"✅ Search completed")
            print(f"Total results: {len(results)}")
            
            # Check each result
            has_reviews_data = 0
            has_analysis = 0
            
            for i, result in enumerate(results[:5]):
                name = result.get('name', 'Unknown')
                print(f"\n{i+1}. {name}")
                
                # Check for reviews_data (should NOT be present)
                if 'reviews_data' in result:
                    has_reviews_data += 1
                    print(f"   ❌ ERROR: reviews_data found (size: {len(result['reviews_data'])} reviews)")
                else:
                    print(f"   ✅ reviews_data correctly excluded")
                
                # Check for reviews_analysis (should be present if eligible)
                if 'reviews_analysis' in result:
                    has_analysis += 1
                    analysis = result['reviews_analysis']
                    if analysis:  # Check if analysis is not None
                        print(f"   ✅ AI analysis present:")
                        print(f"      - Status: {analysis.get('analysis_status', 'N/A')}")
                        print(f"      - Sentiment: {analysis.get('overall_sentiment', 'N/A')}")
                        print(f"      - Analyzed: {analysis.get('total_reviews_analyzed', 0)} reviews")
                    else:
                        print(f"   ⚠️  AI analysis is None")
                else:
                    reviews = result.get('reviews', 0)
                    if reviews >= 10:
                        print(f"   ⚠️  No analysis (but has {reviews} reviews)")
                    else:
                        print(f"   ℹ️  No analysis ({reviews} reviews < minimum)")
            
            print(f"\n📊 Summary:")
            print(f"Results with reviews_data: {has_reviews_data} (should be 0)")
            print(f"Results with AI analysis: {has_analysis}")
            
            # Save sample output
            with open('test_no_reviews_output.json', 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n💾 Full output saved to test_no_reviews_output.json")
            
            # Check file size difference
            import os
            size = os.path.getsize('test_no_reviews_output.json')
            print(f"File size: {size:,} bytes ({size/1024:.1f} KB)")
            
            return data
            
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"\n❌ Exception: {str(e)}")
        return None

if __name__ == "__main__":
    test_no_reviews_data()