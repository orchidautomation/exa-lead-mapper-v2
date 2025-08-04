#!/usr/bin/env python3
"""
Test script to verify Groq structured outputs work correctly
"""
import os
import json
from groq import Groq

# Test with different models
MODELS_TO_TEST = [
    ("moonshotai/kimi-k2-instruct", True),  # Supports structured outputs
    ("llama-3.3-70b-versatile", False),     # JSON Object Mode only
]

def test_structured_output():
    """Test structured output with Groq models."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    test_text = """
    Review #1 (5/5 stars): The golf course is immaculate! The grounds crew does an amazing job maintaining the fairways and greens. Worth every penny.
    Review #2 (4/5 stars): Great course, well-maintained. The staff is very friendly. Only downside is it gets crowded on weekends.
    Review #3 (5/5 stars): Beautiful course with excellent maintenance. The greens are always in perfect condition.
    """
    
    for model_name, supports_structured in MODELS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"Supports structured outputs: {supports_structured}")
        print(f"{'='*60}")
        
        try:
            if supports_structured:
                # Test with structured outputs using strict=True format
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": f"Analyze these golf course reviews and extract insights:\n{test_text}"
                    }],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "review_analysis",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "overall_sentiment": {
                                        "type": "string",
                                        "enum": ["positive", "negative", "mixed"]
                                    },
                                    "key_insights": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "maintenance_quality": {
                                        "type": "string",
                                        "enum": ["excellent", "good", "average", "poor"]
                                    }
                                },
                                "required": ["overall_sentiment", "key_insights", "maintenance_quality"],
                                "additionalProperties": False
                            }
                        }
                    },
                    max_tokens=500,
                    temperature=0.1
                )
            else:
                # Test with JSON Object Mode
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": f"""Analyze these golf course reviews and return JSON with:
- overall_sentiment: positive/negative/mixed
- key_insights: array of insights
- maintenance_quality: excellent/good/average/poor

Reviews:
{test_text}

Return only valid JSON."""
                    }],
                    response_format={"type": "json_object"},
                    max_tokens=500,
                    temperature=0.1
                )
            
            result = json.loads(response.choices[0].message.content)
            print(f"✅ Success! Got valid JSON:")
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment")
        print("Run: export GROQ_API_KEY='your-key-here'")
        exit(1)
    
    test_structured_output()