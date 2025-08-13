"""
🚀 Quantum Discovery - Modular Architecture
==========================================
Embeddings → Quantum → Display → Gemini Post-Processing

The perfect balance: Ultra-fast initial results with intelligent refinement
"""

import streamlit as st
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import time

# Import our modular components
from components.result_card import render_result_card
from components.pipeline_status import render_pipeline_status
from search.embedding_search import phase1_embedding_search
from search.quantum_refinement import phase2_quantum_refinement
from validation.gemini_validator import phase3_gemini_validation_stream

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Configure page
st.set_page_config(
    page_title="🌌 Quantum Discovery (Modular)",
    page_icon="⚡",
    layout="wide"
)

# Professional dark theme CSS
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(100, 255, 218, 0.5);
    }
    
    /* Result cards */
    .result-card {
        background: #2d2d44;
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .result-card.loading {
        opacity: 0.7;
    }
    
    .result-card.validated {
        border: 2px solid #51cf66;
        background: rgba(81, 207, 102, 0.05);
    }
    
    .result-card.rejected {
        display: none !important;
    }
    
    .result-name {
        color: #64ffda !important;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .result-title {
        color: #ffffff !important;
        margin-bottom: 1rem;
    }
    
    .skill-pill {
        display: inline-block;
        background: rgba(100, 255, 218, 0.2);
        color: #64ffda !important;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
        border: 1px solid rgba(100, 255, 218, 0.3);
    }
    
    .quantum-score {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .gemini-explanation {
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #64ffda;
        border-radius: 5px;
        font-style: italic;
    }
    
    .pipeline-status {
        background: rgba(0, 0, 0, 0.3);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .pipeline-step {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .pipeline-step.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize MongoDB connection
@st.cache_resource
def init_mongodb():
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[os.getenv("DB_NAME", "MagicCRM")]
    return db

db = init_mongodb()


def get_profiles_by_depth(search_depth, selected_userId):
    """
    Get profiles based on search depth selection
    """
    if "Public" in search_depth:
        return list(db["public_profiles"].find(
            {},
            {"embedding": 1, "name": 1, "slug": 1, "areasOfNetworkStrength": 1, 
             "blurb": 1, "title": 1, "company": 1}
        ).limit(5000))
        
    elif "1st degree" in search_depth:
        # Step 1: Get user's direct connections from private_profiles (just slugs)
        private_connections = list(db["private_profiles"].find(
            {"userId": selected_userId},
            {"slug": 1}
        ).limit(1000))
        
        # Step 2: Extract slugs
        connection_slugs = [p.get('slug') for p in private_connections if p.get('slug')]
        
        # Step 3: Fetch actual profiles from public_profiles (protects PII)
        if connection_slugs:
            return list(db["public_profiles"].find(
                {"slug": {"$in": connection_slugs}},
                {"embedding": 1, "name": 1, "slug": 1, "areasOfNetworkStrength": 1,
                 "blurb": 1, "title": 1, "company": 1}
            ))
        return []
        
    else:  # 2nd degree connections (Friend-of-Friend)
        # Step 1: Get user's direct connections and their slugs
        first_degree = list(db["private_profiles"].find(
            {"userId": selected_userId},
            {"slug": 1}
        ).limit(500))
        
        # Step 2: Extract slugs to use as userIds for FoF search
        friend_slugs = [profile.get('slug') for profile in first_degree if profile.get('slug')]
        
        # Step 3: Get friends of friends (2nd degree)
        second_degree_slugs = set()
        if friend_slugs:
            # Find 2nd degree connections in private_profiles
            second_degree = list(db["private_profiles"].find(
                {"userId": {"$in": friend_slugs}},
                {"slug": 1}
            ).limit(3000))
            
            # Collect all slugs (1st + 2nd degree)
            for profile in second_degree:
                if profile.get('slug'):
                    second_degree_slugs.add(profile.get('slug'))
            
            # Also add 1st degree slugs
            for slug in friend_slugs:
                second_degree_slugs.add(slug)
        
        # Step 4: Fetch actual profiles from public_profiles (protects PII)
        if second_degree_slugs:
            return list(db["public_profiles"].find(
                {"slug": {"$in": list(second_degree_slugs)}},
                {"embedding": 1, "name": 1, "slug": 1, "areasOfNetworkStrength": 1,
                 "blurb": 1, "title": 1, "company": 1}
            ))
        return []


# Main UI
st.markdown("""
<div class="main-header">
    <h1>⚡ Quantum Discovery Pipeline</h1>
    <p>Modular Architecture for Scalable Search</p>
</div>
""", unsafe_allow_html=True)

# Search interface
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🎯 Search Settings")
    
    # User selection
    users_list = list(db["users"].find({}, {"userId": 1, "name": 1, "_id": 0}))
    if users_list:
        user_options = {f"{u.get('name', 'Unknown')} ({u['userId']})": u['userId'] 
                       for u in users_list if 'userId' in u}
        selected_user = st.selectbox("Select User", options=list(user_options.keys()))
        selected_userId = user_options[selected_user]
    
    search_depth = st.radio(
        "Search Depth",
        ["1st degree connections", "2nd degree connections", "🌍 ALL Public Profiles"],
        index=2  # Default to all public
    )
    
    result_limit = st.slider("Number of Results", 5, 20, 10)

with col2:
    st.markdown("### 🔍 Search Query")
    
    search_query = st.text_area(
        "What are you looking for?",
        placeholder="songwriter, AI researcher, investment partner, creative director...",
        height=100,
        label_visibility="collapsed"
    )
    
    if st.button("🚀 Launch Quantum Search", type="primary", use_container_width=True):
        if search_query:
            
            # Create containers for dynamic updates
            pipeline_placeholder = st.empty()
            results_container = st.container()
            
            # Track times for pipeline status
            pipeline_times = {}
            
            # Initial pipeline status
            pipeline_placeholder.markdown(
                render_pipeline_status("embeddings", pipeline_times), 
                unsafe_allow_html=True
            )
            
            # Phase 1: Get profiles based on search depth
            start_time = time.time()
            all_profiles = get_profiles_by_depth(search_depth, selected_userId)
            
            # Phase 1: Embedding search
            candidates = phase1_embedding_search(
                search_query, 
                all_profiles, 
                openai_client,
                limit=result_limit*3
            )
            pipeline_times['embeddings'] = time.time() - start_time
            
            # Update pipeline status
            pipeline_placeholder.markdown(
                render_pipeline_status("quantum", pipeline_times),
                unsafe_allow_html=True
            )
            
            # Phase 2: Quantum refinement
            phase2_start = time.time()
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=search_query,
                dimensions=1536
            )
            query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            results = phase2_quantum_refinement(query_embedding, candidates, limit=result_limit)
            pipeline_times['quantum'] = time.time() - phase2_start
            
            # Update pipeline status
            pipeline_placeholder.markdown(
                render_pipeline_status("display", pipeline_times),
                unsafe_allow_html=True
            )
            
            # Phase 3: Display results immediately
            with results_container:
                st.markdown("---")
                st.markdown("### 🎯 Search Results")
                
                # Create placeholders for each result
                result_placeholders = []
                for i, profile in enumerate(results):
                    placeholder = st.empty()
                    
                    # Display initial result
                    placeholder.markdown(
                        render_result_card(profile, i, status="loading"),
                        unsafe_allow_html=True
                    )
                    
                    result_placeholders.append((placeholder, profile))
            
            pipeline_times['display'] = 0  # Instant
            
            # Update pipeline status for validation phase
            pipeline_placeholder.markdown(
                render_pipeline_status("validation", pipeline_times),
                unsafe_allow_html=True
            )
            
            # Phase 4: Sequential Gemini validation with streaming
            validated_results = []
            for i, (placeholder, profile) in enumerate(result_placeholders):
                stream = phase3_gemini_validation_stream(search_query, profile, gemini_model)
                
                if not stream:
                    continue
                
                full_response = ""
                explanation_html = ""
                
                # Build skills HTML once
                skills_html = ""
                if profile.get('areasOfNetworkStrength'):
                    skills_html = '<div>' + ''.join([
                        f'<span class="skill-pill">{s}</span>' 
                        for s in profile['areasOfNetworkStrength'][:5]
                    ]) + '</div>'
                
                # Stream the response
                for chunk in stream:
                    if chunk.text:
                        full_response += chunk.text
                        # Stream the explanation part
                        if "REASON:" in full_response:
                            reason_text = full_response.split("REASON:")[-1].strip()
                            explanation_html = f'<div class="gemini-explanation" id="gemini-explanation-{i}">{reason_text}</div>'
                            
                            # Update with streaming content
                            placeholder.markdown(
                                render_result_card(
                                    profile, 
                                    i, 
                                    status="loading",
                                    explanation=explanation_html
                                ),
                                unsafe_allow_html=True
                            )
                        time.sleep(0.01)  # Small delay for smooth streaming
                
                # Check if it's a match
                if "MATCH: YES" in full_response:
                    profile['explanation_html'] = explanation_html
                    validated_results.append(profile)
            
            # Phase 5: Re-render the validated results cleanly
            # Clear the old placeholders
            for placeholder, _ in result_placeholders:
                placeholder.empty()
            
            # Display only the validated results
            with results_container:
                for i, profile in enumerate(validated_results):
                    st.markdown(
                        render_result_card(
                            profile,
                            i,
                            status="validated",
                            explanation=profile.get('explanation_html', '')
                        ),
                        unsafe_allow_html=True
                    )
            
            # Final pipeline status update
            pipeline_times['total'] = time.time() - start_time
            pipeline_placeholder.markdown(
                render_pipeline_status("complete", pipeline_times),
                unsafe_allow_html=True
            )

# Footer
st.markdown("---")
st.caption("⚡ Quantum Superiority: Modular Architecture Edition")