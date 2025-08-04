#!/usr/bin/env python3
"""
Simple test script to verify the setup works correctly.
"""

import os
import sys
import requests
import json
from datetime import datetime

def test_import():
    """Test if all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from config.settings import settings
        print("✅ Config imported successfully")
        
        from models.schemas import SearchRequest, PlaceResult
        print("✅ Models imported successfully")
        
        from services.database import DatabaseManager
        print("✅ Database service imported successfully")
        
        from services.geocoding import GeocodingFactory
        print("✅ Geocoding service imported successfully")
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_database():
    """Test database initialization."""
    print("\n🗄️ Testing database...")
    
    try:
        from services.database import DatabaseManager
        
        # Test with in-memory database
        db = DatabaseManager('sqlite:///:memory:')
        
        # Test API key creation
        test_key = "test_key_123"
        db.create_api_key(test_key, "Test key")
        
        # Test API key validation
        if db.validate_api_key(test_key):
            print("✅ Database operations successful!")
            return True
        else:
            print("❌ API key validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False

def test_geocoding():
    """Test geocoding service."""
    print("\n🌍 Testing geocoding...")
    
    try:
        from services.database import DatabaseManager
        from services.geocoding import GeocodingFactory
        
        db = DatabaseManager('sqlite:///:memory:')
        geocoding = GeocodingFactory.create('open-meteo', db)
        
        # Test geocoding a simple location
        result = geocoding.geocode("New York, NY")
        
        if result and len(result) == 2:
            lat, lon = result
            print(f"✅ Geocoding successful: New York, NY -> ({lat}, {lon})")
            return True
        else:
            print("❌ Geocoding returned invalid result")
            return False
            
    except Exception as e:
        print(f"❌ Geocoding test failed: {str(e)}")
        return False

def test_server_start():
    """Test if the server can start."""
    print("\n🚀 Testing server startup...")
    
    try:
        # Import Flask app
        from app import create_app
        
        app = create_app()
        
        # Test app creation
        if app:
            print("✅ Flask app created successfully!")
            
            # Test a simple route
            with app.test_client() as client:
                response = client.get('/')
                if response.status_code == 200:
                    print("✅ Root endpoint accessible!")
                    return True
                else:
                    print(f"❌ Root endpoint returned {response.status_code}")
                    return False
        else:
            print("❌ Failed to create Flask app")
            return False
            
    except Exception as e:
        print(f"❌ Server startup test failed: {str(e)}")
        return False

def test_health_endpoint():
    """Test health endpoint without external dependencies."""
    print("\n🏥 Testing health endpoint...")
    
    try:
        from app import create_app
        
        app = create_app()
        
        with app.test_client() as client:
            response = client.get('/api/v1/health')
            
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Health endpoint returned: {data.get('status', 'unknown')}")
                return True
            else:
                print(f"❌ Health endpoint returned {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Health endpoint test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🧪 Business Mapper API v2.0 - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("Database Test", test_database),
        ("Geocoding Test", test_geocoding),
        ("Server Startup Test", test_server_start),
        ("Health Endpoint Test", test_health_endpoint)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is working correctly.")
        print("\n🚀 Ready to start the server with: python app.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)