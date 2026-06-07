"""
PRISM Anthropic-to-Gemini API compatibility wrapper.
If the API key is a Gemini API key (starts with 'AIzaSy') or GEMINI_API_KEY is configured,
this wrapper translates Claude API calls to Gemini API calls.
Otherwise, it dynamically delegates to the standard installed 'anthropic' SDK.
"""

import sys
import os
import requests
import json

# Helper to load the real installed anthropic library if needed
def get_real_anthropic():
    original_path = sys.path[:]
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Remove workspace dir from sys.path to avoid self-reference
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(workspace_dir)]
    
    # Temporarily remove wrapper from sys.modules to force loading real package
    cached_wrapper = sys.modules.pop("anthropic", None)
    try:
        import anthropic as real
        return real
    finally:
        # Restore sys.path and cached wrapper
        sys.path = original_path
        if cached_wrapper:
            sys.modules["anthropic"] = cached_wrapper

class MessageContent:
    def __init__(self, text):
        self.text = text

class MessageResponse:
    def __init__(self, text):
        self.content = [MessageContent(text)]

class GeminiMessages:
    def __init__(self, api_key):
        self.api_key = api_key

    def create(self, model, max_tokens, messages, system=None, temperature=0.0, **kwargs):
        # Map Claude models to Gemini 1.5 Flash (incredibly cheap/fast/highly capable)
        gemini_model = "gemini-1.5-flash"
        
        # Format messages for Gemini's contents structure
        gemini_contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Gemini requires roles to be 'user' or 'model' ('assistant' is rejected)
            if role == "assistant":
                role = "model"
            gemini_contents.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}]
            })
            
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        # Handle system instructions
        if system:
            payload["systemInstruction"] = {
                "parts": [{"text": system}]
            }
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
        
        try:
            text_result = data["candidates"][0]["content"]["parts"][0]["text"]
            return MessageResponse(text_result)
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response structure from Gemini API: {data}")

class Anthropic:
    def __init__(self, api_key=None, **kwargs):
        # Resolve API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
        # Check if the key is a Gemini API key
        is_gemini = False
        if self.api_key and (self.api_key.startswith("AIzaSy") or os.environ.get("GEMINI_API_KEY")):
            is_gemini = True
            
        if is_gemini:
            # Use Gemini wrapper
            gemini_key = self.api_key
            if not gemini_key and os.environ.get("GEMINI_API_KEY"):
                gemini_key = os.environ.get("GEMINI_API_KEY")
            self.messages = GeminiMessages(gemini_key)
            self._mode = "gemini"
        else:
            # Delegate to standard Anthropic SDK
            real_anthropic = get_real_anthropic()
            self._delegate = real_anthropic.Anthropic(api_key=self.api_key, **kwargs)
            self.messages = self._delegate.messages
            self._mode = "anthropic"

# Re-export exceptions from real anthropic for compatibility
try:
    real_lib = get_real_anthropic()
    APIError = real_lib.APIError
    APIConnectionError = real_lib.APIConnectionError
    APITimeoutError = real_lib.APITimeoutError
    AuthenticationError = real_lib.AuthenticationError
    PermissionDeniedError = real_lib.PermissionDeniedError
    NotFoundError = real_lib.NotFoundError
    RateLimitError = real_lib.RateLimitError
except Exception:
    class APIError(Exception): pass
    class APIConnectionError(APIError): pass
    class APITimeoutError(APIError): pass
    class AuthenticationError(APIError): pass
    class PermissionDeniedError(APIError): pass
    class NotFoundError(APIError): pass
    class RateLimitError(APIError): pass
