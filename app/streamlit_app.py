"""
Enhanced Streamlit Application for Customer Feedback Analysis
Advanced UI with analytics dashboard and maximum accuracy models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from utils import TextPreprocessor

# Action Categories Logic
ACTION_CATEGORIES = {
    'Food': ['Inspect Kitchen Hygiene', 'Revise Menu Quality', 'Staff Training (Food)', 'Compensate Guest'],
    'Rooms': ['Deep Clean & Maintenance', 'Upgrade Room Amenities', 'Pest Control Investigation'],
    'Services': ['Staff Communication Training', 'Review Check-in Process', 'Disciplinary Action'],
    'Recreation': ['Pool Maintenance', 'Update Gym Equipment', 'Recreation Staff Review']
}

# Set page config
st.set_page_config(
    page_title="Advanced Customer Feedback Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS - Premium Design System
st.markdown("""
<style>
    /* 1. Global Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* 2. Main Container & Background */
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(circle at 50% 0%, #1f1f3a 0%, #0e1117 70%);
        background-attachment: fixed;
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding-top: 1rem;
        text-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    }
    
    /* 3. Cards & Containers */
    div[data-testid="stMetric"], .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    label[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* 4. Action Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary Buttons */
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: none;
    }
    
    /* 5. Inputs & Selectboxes */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > div, 
    .stTextArea > div > div > textarea {
        background-color: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(75, 85, 99, 0.4);
        color: #f3f4f6;
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input:focus, 
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* 6. Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.03);
        border-radius: 10px;
        color: #cbd5e1;
        font-weight: 500;
        padding: 0 20px;
        border: 1px solid transparent;
        transition: all 0.3s;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.15) !important;
        border: 1px solid rgba(99, 102, 241, 0.5) !important;
        color: #a5b4fc !important;
        font-weight: 700;
    }
    
    /* 7. Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19; /* Very dark blue/black */
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* 8. Success/Error/Info Messages */
    .stAlert {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(5px);
    }
    
    div[data-testid="stNotification"] {
        background-color: #1e1e2e;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

</style>
""", unsafe_allow_html=True)

# Get absolute paths
BASE_DIR = Path(__file__).parent.parent


@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    errors = []
    
    # Sentiment Model
    sentiment_path = BASE_DIR / 'models/sentiment_analyzer/best_sentiment_model.pkl'
    try:
        if sentiment_path.exists():
            with open(sentiment_path, 'rb') as f:
                models['sentiment'] = pickle.load(f)
            st.sidebar.success("✅ Sentiment Model Loaded")
        else:
            errors.append("Sentiment model not found")
    except Exception as e:
        errors.append(f"Sentiment model error: {e}")
    
    # Category Model - Try new one first, fallback to legacy
    category_paths = [
        BASE_DIR / 'models/category_classifier/best_category_model.pkl',
        BASE_DIR / 'models/category_classifier_model.pkl'  # Legacy
    ]
    
    for category_path in category_paths:
        try:
            if category_path.exists():
                with open(category_path, 'rb') as f:
                    models['category'] = pickle.load(f)
                st.sidebar.success(f"✅ Category Model Loaded")
                break
        except Exception as e:
            continue
    
    if 'category' not in models:
        st.sidebar.warning("⚠️ Category model not loaded - manual selection required")
    
    # Recommendation Model
    rec_path = BASE_DIR / 'models/recommendation_engine/best_recommendation_model.pkl'
    try:
        if rec_path.exists():
            with open(rec_path, 'rb') as f:
                models['recommendation'] = pickle.load(f)
            st.sidebar.success("✅ Recommendation Model Loaded")
        else:
            errors.append("Recommendation model not found")
    except Exception as e:
        errors.append(f"Recommendation model error: {e}")
        
    # Label Encoder (New)
    le_path = BASE_DIR / 'models/recommendation_engine/action_label_encoder.pkl'
    try:
        if le_path.exists():
            with open(le_path, 'rb') as f:
                models['label_encoder'] = pickle.load(f)
        else:
            # Silent warning to avoid clutter, or log it
            pass
    except Exception as e:
        errors.append(f"Label encoder error: {e}")
    
    return models, errors


# Initialize
models, loading_errors = load_models()
preprocessor = TextPreprocessor()

# Header
st.markdown('<h1 class="main-header">🏨 Customer Feedback Analysis</h1>', unsafe_allow_html=True)

# Sidebar - Model Info
with st.sidebar:
    # Minimal Sidebar
    st.markdown("### Status")
    for model_name in models:
        st.success(f"✅ {model_name.replace('_', ' ').title()}")


# Main tabs
tab1, tab2 = st.tabs(["📝 Analysis", "📊 Batch Processing"])

# TAB 1: Single Review
with tab1:
    st.header("Analyze Individual Review")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        review_text = st.text_area(
            "Enter customer feedback:",
            height=150,
            placeholder="e.g., The room was dirty and the staff was very rude. The food quality was also poor..."
        )
    
    with col2:
        auto_detect = st.checkbox("Auto-detect category", value=True)
        
        if not auto_detect:
            category = st.selectbox(
                "Select Category:",
                ["Food", "Rooms", "Services", "Recreation"]
            )
        else:
            category = None
        
        analyze_btn = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
    
    if analyze_btn and review_text:
        if 'sentiment' not in models:
            st.error("❌ Sentiment model not loaded")
        else:
            with st.spinner("Analyzing..."):
                # Get sentiment (Advanced Logic)
                preprocessor = TextPreprocessor()
                cleaned_review = preprocessor.clean_text(review_text)
                
                sentiment_model = models['sentiment']
                conf_score = 1.0
                neg_score = 0.0
                
                if isinstance(sentiment_model, dict):
                    # Advanced Model Pipeline
                    tfidf_feat = sentiment_model['vectorizer'].transform([cleaned_review]).toarray()
                    adv_feat = preprocessor.extract_advanced_features(review_text)
                    X_input = np.hstack([tfidf_feat, adv_feat])
                    
                    sentiment = sentiment_model['model'].predict(X_input)[0]
                    
                    if hasattr(sentiment_model['model'], 'predict_proba'):
                        probs = sentiment_model['model'].predict_proba(X_input)[0]
                        classes = sentiment_model['model'].classes_
                        conf_score = probs[np.where(classes == sentiment)[0][0]]
                        if 'negative' in classes:
                            neg_score = probs[np.where(classes == 'negative')[0][0]]
                else:
                    # Simple Model Fallback
                    sentiment = sentiment_model.predict([cleaned_review])[0]
                    neg_score = 1.0 if sentiment == 'negative' else 0.0
                
                # Display results
                st.divider()
                st.subheader("Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if sentiment == 'negative':
                        st.error(f"**Sentiment:** {sentiment.upper()}")
                        sentiment_emoji = "😞"
                    elif sentiment == 'positive':
                        st.success(f"**Sentiment:** {sentiment.upper()}")
                        sentiment_emoji = "😊"
                    else:
                        st.info(f"**Sentiment:** {sentiment.upper()}")
                        sentiment_emoji = "😐"
                    
                    st.caption(f"**Confidence:** {conf_score:.1%}")
                    if sentiment == 'negative':
                        st.caption(f"**Negative Severity:** {neg_score:.1%}")
                
                # Auto-detect category if needed
                if auto_detect and 'category' in models:
                    detected_category = models['category'].predict([cleaned_review])[0]
                    with col2:
                        st.info(f"**Category:** {detected_category}")
                    category = detected_category
                elif auto_detect:
                    category = st.selectbox(
                        "Select Category (auto-detection unavailable):",
                        ["Food", "Rooms", "Services", "Recreation"]
                    )
                    with col2:
                        st.warning(f"**Category:** {category} (Manual)")
                else:
                    with col2:
                        st.info(f"**Category:** {category}")
                
                # Get recommendation for negative reviews
                if sentiment == 'negative' and category and 'recommendation' in models:
                    rec_model = models['recommendation']
                    
                    # 1. Get raw probabilities
                    try:
                        probs = rec_model.predict_proba([cleaned_review])[0]
                        classes = rec_model.classes_ # These are integers [0, 1, 2...]
                        
                        # 2. Apply Category Masking (Constrained Decoding)
                        if 'label_encoder' in models:
                            le = models['label_encoder']
                            
                            # Valid actions for the selected category
                            valid_actions = ACTION_CATEGORIES.get(category, [])
                            
                            if valid_actions:
                                # Find indices of valid actions
                                valid_indices = []
                                for act in valid_actions:
                                    try:
                                        # Convert text action to integer label
                                        lbl = le.transform([act])[0]
                                        # Find where this label is in the classes array
                                        idx = np.where(classes == lbl)[0]
                                        if len(idx) > 0:
                                            valid_indices.append(idx[0])
                                    except:
                                        continue
                                
                                if valid_indices:
                                    # Create mask
                                    mask = np.zeros_like(probs)
                                    mask[valid_indices] = 1.0
                                    
                                    # Apply mask and re-normalize
                                    masked_probs = probs * mask
                                    
                                    # Explain logic if needed (debug)
                                    # st.write(f"Masked Probs: {masked_probs}")
                                    
                                    if masked_probs.sum() > 0:
                                        best_idx = np.argmax(masked_probs)
                                        predicted_label = classes[best_idx]
                                        action = le.inverse_transform([predicted_label])[0]
                                    else:
                                        # Fallback if no valid overlap
                                        action = rec_model.predict([cleaned_review])[0]
                                        action = le.inverse_transform([action])[0]
                                else:
                                    # Fallback if validation fails
                                    action = rec_model.predict([cleaned_review])[0]
                                    action = le.inverse_transform([action])[0]
                            else:
                                # No specific mapping for this category
                                action = rec_model.predict([cleaned_review])[0]
                                action = le.inverse_transform([action])[0]
                        else:
                            # No label encoder, just predict raw
                            action = rec_model.predict([cleaned_review])[0]
                    except Exception as e:
                        # Fallback on error
                        # st.error(f"Prediction logic error: {e}")
                        action = rec_model.predict([cleaned_review])[0]
                        if 'label_encoder' in models:
                             action = models['label_encoder'].inverse_transform([action])[0]
                    
                    st.markdown(f"""
                    <div style="background: rgba(255, 75, 75, 0.1); border: 1px solid rgba(255, 75, 75, 0.3); border-radius: 12px; padding: 20px;">
                        <h2 style="color: #ff4b4b; margin:0;">🎯 {action}</h2>
                        <p style="color: #ecc; margin-top: 10px; font-size: 0.9em;">
                            <b>Category:</b> {category} | <b>Confidence:</b> {conf_score:.0%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if sentiment == 'positive':
                        st.balloons()
                        st.success("✅  No action needed - positive feedback!")
                    elif sentiment == 'neutral':
                        st.info("ℹ️ Neutral feedback - monitoring recommended")
    
    elif analyze_btn:
        st.warning("Please enter review text")

# TAB 2: Batch Processing
with tab2:
    st.header("Batch Process Multiple Reviews")
    
    st.info("""
    **Upload a CSV file with the following structure:**
    - `review` or `text` column: The review text
    - `category` column (optional): Food/Rooms/Services/Recreation
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.write("**Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Detect columns
            text_col = None
            for col in ['review', 'text', 'Review_text', 'cleaned_text']:
                if col in df.columns:
                    text_col = col
                    break
            
            cat_col = 'category' if 'category' in df.columns else None
            
            if not text_col:
                st.error("❌ No review text column found")
            elif 'sentiment' not in models:
                st.error("❌ Sentiment model not loaded")
            else:
                process_btn = st.button("🚀 Process All Reviews", type="primary")
                
                if process_btn:
                    with st.spinner(f"Processing {len(df)} reviews..."):
                        progress_bar = st.progress(0)
                        
                        # Clean text
                        df['cleaned_text'] = df[text_col].fillna('').apply(
                            lambda x: preprocessor.clean_text(x)
                        )
                        progress_bar.progress(0.2)
                        
                        # Sentiment analysis
                        df['sentiment'] = models['sentiment'].predict(df['cleaned_text'])
                        progress_bar.progress(0.5)
                        
                        # Category detection/assignment
                        if not cat_col and 'category' in models:
                            df['category'] = models['category'].predict(df['cleaned_text'])
                        elif not cat_col:
                            default_cat = st.selectbox("Select default category:", 
                                                      ["Food", "Rooms", "Services", "Recreation"])
                            df['category'] = default_cat
                        else:
                            df['category'] = df[cat_col]
                        
                        progress_bar.progress(0.7)
                        
                        # Recommendations for negative reviews
                        df['recommended_action'] = ''
                        negative_mask = df['sentiment'] == 'negative'
                        
                        if negative_mask.any() and 'recommendation' in models:
                            negative_df = df[negative_mask]
                            input_data = negative_df[['cleaned_text', 'category']]
                            actions = models['recommendation'].predict(input_data)
                            df.loc[negative_mask, 'recommended_action'] = actions
                        
                        progress_bar.progress(1.0)
                        
                        st.success(f"✅ Processed {len(df)} reviews!")
                        
                        # Results
                        st.subheader("Results")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Reviews", len(df))
                        with col2:
                            neg_count = (df['sentiment'] == 'negative').sum()
                            st.metric("Negative", neg_count, delta=f"-{neg_count/len(df)*100:.1f}%")
                        with col3:
                            pos_count = (df['sentiment'] == 'positive').sum()
                            st.metric("Positive", pos_count, delta=f"+{pos_count/len(df)*100:.1f}%")
                        with col4:
                            neu_count = (df['sentiment'] == 'neutral').sum()
                            st.metric("Neutral", neu_count)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Sentiment distribution
                            fig1 = px.pie(
                                df, 
                                names='sentiment', 
                                title='Sentiment Distribution',
                                color='sentiment',
                                color_discrete_map={'positive': '#00CC96', 'negative': '#EF553B', 'neutral': '#636EFA'}
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            # Category distribution
                            fig2 = px.bar(
                                df['category'].value_counts().reset_index(),
                                x='category',
                                y='count',
                                title='Feedback by Category',
                                labels={'category': 'Category', 'count': 'Count'},
                                color='count',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Data table
                        st.dataframe(df, use_container_width=True)
                        
                        # Download
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="analyzed_reviews.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer removed
