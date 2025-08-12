"""
Pipeline Status Component for Quantum Discovery
"""

def render_pipeline_status(step="embeddings", times=None):
    """
    Renders the pipeline status bar
    
    Args:
        step: Current active step
        times: Dictionary of completion times for each step
    
    Returns:
        HTML string for the pipeline status
    """
    times = times or {}
    
    # Define steps and their states
    steps = {
        "embeddings": {"icon": "📡", "label": "Embeddings"},
        "quantum": {"icon": "⚛️", "label": "Quantum"},
        "display": {"icon": "📊", "label": "Display"},
        "validation": {"icon": "🤖", "label": "Validation"},
        "complete": {"icon": "✅", "label": "Complete"}
    }
    
    # Build status HTML
    if step == "complete":
        total_time = times.get('total', 0)
        status_html = f"""
        <div class="pipeline-status">
            <span class="pipeline-step">✅ Complete Pipeline: {total_time:.2f}s</span>
        </div>
        """
    else:
        step_elements = []
        for step_key, step_info in steps.items():
            if step_key == "complete":
                continue
                
            # Determine step state
            if step_key in times:
                # Completed step
                time_str = f" ({times[step_key]:.2f}s)" if times[step_key] else ""
                step_html = f'<span class="pipeline-step">✅ {step_info["label"]}{time_str}</span>'
            elif step_key == step:
                # Current active step
                step_html = f'<span class="pipeline-step active">{step_info["icon"]} {step_info["label"]}</span>'
            else:
                # Pending step
                step_html = f'<span class="pipeline-step">{step_info["icon"]} {step_info["label"]}</span>'
            
            step_elements.append(step_html)
        
        status_html = f"""
        <div class="pipeline-status">
            {' '.join(step_elements)}
        </div>
        """
    
    return status_html