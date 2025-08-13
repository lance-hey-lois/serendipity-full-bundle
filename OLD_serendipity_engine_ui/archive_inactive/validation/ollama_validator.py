"""
Ollama Validation Module - Fallback for when Gemini API is unavailable
"""

import requests
import json
from typing import Generator, Optional, Dict, Any


def validate_with_ollama_stream(query: str, profile: dict, model: str = "gemma3:4b") -> Optional[Generator]:
    """
    Validate a profile using local Ollama instance with streaming
    
    Args:
        query: Search query string
        profile: Profile dictionary to validate
        model: Ollama model to use (default: gemma3:4b for speed)
    
    Returns:
        Generator yielding response chunks or None on error
    """
    validation_prompt = f"""
    Evaluate if this profile truly matches the search query.
    
    SEARCH QUERY: "{query}"
    
    PROFILE:
    Name: {profile.get('name', 'Unknown')}
    Title: {profile.get('title', 'N/A')}
    Company: {profile.get('company', 'N/A')}
    Skills: {', '.join(profile.get('areasOfNetworkStrength', [])[:5])}
    Bio: {profile.get('blurb', '')[:300]}
    
    Respond with ONLY:
    MATCH: YES or NO
    CONFIDENCE: 0-100
    REASON: Brief explanation why this is or isn't a match
    
    Be REASONABLE - return YES if there's a plausible connection to the query.
    """
    
    try:
        # Ollama API endpoint
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": validation_prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_predict": 150
            }
        }
        
        # Make streaming request
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        def stream_generator():
            """Generator that yields text chunks from Ollama"""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            yield data['response']
                    except json.JSONDecodeError:
                        continue
        
        return stream_generator()
        
    except Exception as e:
        print(f"ERROR in validate_with_ollama_stream: {str(e)}")
        return None


def process_ollama_stream(stream, profile, index):
    """
    Process Ollama streaming response and extract validation result
    
    Args:
        stream: Ollama streaming response generator
        profile: Profile being validated
        index: Result index
    
    Returns:
        Tuple of (is_match, full_response, explanation_text)
    """
    if not stream:
        return False, "", ""
    
    full_response = ""
    
    # Collect the full response
    try:
        for chunk in stream:
            if chunk:
                full_response += chunk
    except Exception as e:
        print(f"Error processing Ollama stream: {e}")
        return False, "", ""
    
    # Parse the response
    match_status = "NO"  # Default to NO
    if "MATCH: YES" in full_response.upper():
        match_status = "YES"
    
    # Extract the reason
    reason_text = ""
    if "REASON:" in full_response.upper():
        # Find the reason line
        lines = full_response.split('\n')
        for i, line in enumerate(lines):
            if "REASON:" in line.upper():
                # Get this line and any following lines
                reason_text = line.split("REASON:", 1)[-1].strip()
                # Add any following lines that might be part of the reason
                for j in range(i+1, min(i+3, len(lines))):
                    if lines[j].strip() and not any(keyword in lines[j].upper() for keyword in ["MATCH:", "CONFIDENCE:"]):
                        reason_text += " " + lines[j].strip()
                break
    
    if not reason_text:
        reason_text = "Unable to extract reason from response"
    
    return match_status == "YES", full_response, reason_text