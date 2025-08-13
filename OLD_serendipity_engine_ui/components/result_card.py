"""
Result Card Component for Quantum Discovery
"""

def render_result_card(profile, index, status="loading", explanation="⏳ Validating match..."):
    """
    Renders a result card with profile information
    
    Args:
        profile: Profile dictionary
        index: Result index
        status: Card status (loading, validated, rejected)
        explanation: Explanation text or HTML
    
    Returns:
        HTML string for the result card
    """
    # Build skills HTML
    skills_html = ""
    if profile.get('areasOfNetworkStrength'):
        skills_html = '<div>' + ''.join([
            f'<span class="skill-pill">{s}</span>' 
            for s in profile['areasOfNetworkStrength'][:5]
        ]) + '</div>'
    
    # Determine explanation HTML
    if isinstance(explanation, str) and not explanation.startswith('<div'):
        explanation_html = f'<div class="gemini-explanation" id="gemini-explanation-{index}">{explanation}</div>'
    else:
        explanation_html = explanation
    
    # Build the complete card HTML
    card_html = f"""
    <div class="result-card {status}" id="result-{index}">
        <div class="quantum-score">Q: {profile.get('final_score', 0):.1%}</div>
        <div class="result-name">{index+1}. {profile.get('name', 'Unknown')}</div>
        <div class="result-title">{profile.get('title', 'N/A')} at {profile.get('company', 'N/A')}</div>
        {skills_html}
        {explanation_html}
    </div>
    """
    
    return card_html