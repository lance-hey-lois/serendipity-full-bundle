"""
Gemini Validation Module for Phase 3/4 of Quantum Discovery Pipeline
"""

import google.generativeai as genai


def phase3_gemini_validation_stream(query: str, profile: dict, gemini_model):
    """
    Phase 3: Gemini validation and explanation with streaming
    
    Args:
        query: Search query string
        profile: Profile dictionary to validate
        gemini_model: Configured Gemini model instance
    
    Returns:
        Streaming response from Gemini or None on error
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
        response = gemini_model.generate_content(
            validation_prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 150,
            },
            stream=True
        )
        
        return response
        
    except Exception as e:
        print(f"ERROR in phase3_gemini_validation_stream: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_gemini_stream(stream, profile, index):
    """
    Process Gemini streaming response and extract validation result
    
    Args:
        stream: Gemini streaming response
        profile: Profile being validated
        index: Result index
    
    Returns:
        Tuple of (is_match, full_response, explanation_html)
    """
    if not stream:
        return False, "", ""
    
    full_response = ""
    explanation_html = ""
    
    # Collect the full response
    for chunk in stream:
        if chunk.text:
            full_response += chunk.text
    
    # Parse the response
    match_status = "NO"  # Default to NO
    if "MATCH: YES" in full_response:
        match_status = "YES"
    
    # Extract the reason
    if "REASON:" in full_response:
        reason_text = full_response.split("REASON:")[-1].strip()
        explanation_html = f'<div class="gemini-explanation" id="gemini-explanation-{index}">{reason_text}</div>'
    
    return match_status == "YES", full_response, explanation_html