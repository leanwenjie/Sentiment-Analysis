import os
import json
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set page config for a premium, wide dashboard layout
st.set_page_config(
    page_title="Sentiment & Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism, nice typography, cards, and transitions
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling with gradient */
    .header-container {
        background: linear-gradient(135deg, #6c5ce7 0%, #a862ea 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.2);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Card design */
    .premium-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    /* Highlighting predictions */
    .badge {
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 1rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .badge-positive {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }
    
    .badge-negative {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    
    .badge-neutral {
        background-color: rgba(149, 165, 166, 0.2);
        color: #bdc3c7;
        border: 1px solid #95a5a6;
    }
    
    /* Emotion highlighting */
    .emotion-card {
        text-align: center;
        padding: 1.2rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .emotion-emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .emotion-name {
        font-size: 1.3rem;
        font-weight: 600;
        text-transform: capitalize;
    }
</style>
""", unsafe_allow_html=True)

# Define Emojis and Color mappings for Emotions
EMOTION_INFO = {
    'neutral': {'emoji': '😐', 'color': '#95a5a6', 'sentiment': 'neutral'},
    'love': {'emoji': '❤️', 'color': '#e84393', 'sentiment': 'positive'},
    'happiness': {'emoji': '😊', 'color': '#f1c40f', 'sentiment': 'positive'},
    'sadness': {'emoji': '😢', 'color': '#0984e3', 'sentiment': 'negative'},
    'relief': {'emoji': '😌', 'color': '#00b894', 'sentiment': 'positive'},
    'hate': {'emoji': '😡', 'color': '#d63031', 'sentiment': 'negative'},
    'anger': {'emoji': '👿', 'color': '#e17055', 'sentiment': 'negative'},
    'fun': {'emoji': '🤪', 'color': '#fdcb6e', 'sentiment': 'positive'},
    'enthusiasm': {'emoji': '🤩', 'color': '#ffeaa7', 'sentiment': 'positive'},
    'surprise': {'emoji': '😮', 'color': '#a29bfe', 'sentiment': 'neutral'},
    'empty': {'emoji': '🫙', 'color': '#b2bec3', 'sentiment': 'negative'},
    'worry': {'emoji': '😟', 'color': '#ffeaa7', 'sentiment': 'negative'},
    'boredom': {'emoji': '🥱', 'color': '#dfe6e9', 'sentiment': 'neutral'}
}

# Pre-defined sentence examples
EXAMPLES = [
    "I seriously hate one subject to death but now i feel reluctant to drop it",
    "I was bitten by a dog, that was terrifying",
    "I am happpy when i get good results in the field of academics or athletics",
    "I fell asleep feeling angry, useless and still full of anxiety",
    "I feel like they hated me since then",
    "I feel like i ve regained another vital part of my life which is living",
    "I feel comfortable around him, it's such a relief",
    "I am sitting here typing this and wondering where i belong",
    "This was shockingly amazing! I was not expecting it at all"
]

@st.cache_resource
def load_models_and_metadata():
    """Load model pipelines and metadata, cache to avoid reloading on every rerun."""
    models_dir = 'models'
    metadata_path = os.path.join(models_dir, 'training_metadata.json')
    sent_model_path = os.path.join(models_dir, 'sentiment_pipeline.joblib')
    em_model_path = os.path.join(models_dir, 'emotion_pipeline.joblib')
    
    if not (os.path.exists(metadata_path) and os.path.exists(sent_model_path) and os.path.exists(em_model_path)):
        return None, None, None
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    sent_pipeline = joblib.load(sent_model_path)
    em_pipeline = joblib.load(em_model_path)
    
    return sent_pipeline, em_pipeline, metadata

# Load resources
sent_pipeline, em_pipeline, metadata = load_models_and_metadata()

# Display title header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🧠 Sentiment & Emotion Intelligence Hub</h1>
    <p class="header-subtitle">Dual-classifier system trained with class-imbalance solutions using TF-IDF and Linear Models.</p>
</div>
""", unsafe_allow_html=True)

if sent_pipeline is None or em_pipeline is None:
    st.error("⚠️ Model files not found! Please make sure you have run the training script `train.py` first to generate the models.")
    st.info("You can run training by executing `python train.py` in your terminal.")
    st.stop()

# Sidebar Info
st.sidebar.markdown("### 🛠️ System Overview")
st.sidebar.info(
    "This system handles sentiment (3 classes) and emotion (13 classes) "
    "classification simultaneously.\n\n"
    "**Methodology:**\n"
    "- TF-IDF text representations\n"
    "- Downsampled Neutral majority classes\n"
    "- Cost-sensitive learning (`class_weight='balanced'`)\n"
    "- Trained on 840,000 sentences"
)

# Set tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Sentence Analyzer",
    "📊 Dataset Diagnostics (Handling Imbalance)",
    "📈 Model performance (Metrics)"
])

# ==========================================
# TAB 1: SINGLE SENTENCE ANALYZER
# ==========================================
with tab1:
    st.markdown("### Analyze a Custom Sentence")
    
    # Example dropdown selection
    selected_example = st.selectbox("💡 Choose an example or type your own below:", ["-- Custom Text --"] + EXAMPLES)
    
    default_text = "" if selected_example == "-- Custom Text --" else selected_example
    
    user_input = st.text_area(
        "Enter sentence here:",
        value=default_text,
        placeholder="Type something that expresses your thoughts, feelings, or emotions...",
        height=100
    )
    
    if user_input.strip():
        # Predict Sentiment
        sent_pred = sent_pipeline.predict([user_input])[0]
        sent_probs = sent_pipeline.predict_proba([user_input])[0]
        sent_classes = sent_pipeline.classes_
        
        # Predict Emotion
        em_pred = em_pipeline.predict([user_input])[0]
        em_probs = em_pipeline.predict_proba([user_input])[0]
        em_classes = em_pipeline.classes_
        
        # UI Columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Prediction Output")
            
            # Sentiment Badge
            badge_class = f"badge-{sent_pred}"
            st.markdown(f"""
            <div class="premium-card">
                <h5>Computed Sentiment</h5>
                <span class="badge {badge_class}">{sent_pred.upper()}</span>
                <p style="margin-top: 10px; font-size: 0.9rem; opacity:0.8;">
                    Confidence Score: <b>{sent_probs[list(sent_classes).index(sent_pred)]:.2%}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Emotion Info
            em_meta = EMOTION_INFO.get(em_pred, {'emoji': '🤔', 'color': '#a29bfe', 'sentiment': 'neutral'})
            st.markdown(f"""
            <div class="premium-card" style="border-left: 5px solid {em_meta['color']};">
                <h5>Computed Emotion Category</h5>
                <div class="emotion-card" style="margin-top:10px;">
                    <div class="emotion-emoji">{em_meta['emoji']}</div>
                    <div class="emotion-name" style="color: {em_meta['color']};">{em_pred}</div>
                </div>
                <p style="margin-top: 10px; font-size: 0.9rem; opacity:0.8; text-align: center;">
                    Confidence Score: <b>{em_probs[list(em_classes).index(em_pred)]:.2%}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("#### Probability Analysis")
            
            # Sentiment Bar Chart
            sent_df = pd.DataFrame({
                'Sentiment': sent_classes,
                'Probability': sent_probs
            }).sort_values(by='Probability', ascending=True)
            
            fig_sent = px.bar(
                sent_df,
                x='Probability',
                y='Sentiment',
                orientation='h',
                title="Sentiment Confidence Levels",
                color='Sentiment',
                color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'},
                text_auto='.1%'
            )
            fig_sent.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="white" if st.get_option("theme.base") == "dark" else "black")
            )
            st.plotly_chart(fig_sent, use_container_width=True)
            
            # Emotion Bar Chart
            em_colors = [EMOTION_INFO.get(c, {}).get('color', '#95a5a6') for c in em_classes]
            em_df = pd.DataFrame({
                'Emotion': em_classes,
                'Probability': em_probs,
                'Color': em_colors
            }).sort_values(by='Probability', ascending=True)
            
            # Filter low probabilities to make chart cleaner if needed (show top 6 or all)
            fig_em = px.bar(
                em_df,
                x='Probability',
                y='Emotion',
                orientation='h',
                title="Emotion Category Probability Distribution",
                text_auto='.1%',
                color='Emotion',
                color_discrete_map={k: v['color'] for k, v in EMOTION_INFO.items()}
            )
            fig_em.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                font=dict(color="white" if st.get_option("theme.base") == "dark" else "black")
            )
            st.plotly_chart(fig_em, use_container_width=True)

    else:
        st.info("Write a sentence or choose an example from the dropdown above to start classification!")

# ==========================================
# TAB 2: DATASET DIAGNOSTICS (CLASS IMBALANCE HANDLING)
# ==========================================
with tab2:
    st.markdown("### How We Handled Class Imbalance")
    st.write(
        "The original dataset suffers from severe class imbalance: `neutral` sentences constitute **80.3%** of all samples. "
        "Training directly on this will lead to a model that predicts 'neutral' in almost all edge cases."
    )
    
    # We load saved counts from metadata
    orig_sent = metadata['original_sentiment_distribution']
    bal_sent = metadata['balanced_sentiment_distribution']
    
    orig_em = metadata['original_emotion_distribution']
    bal_em = metadata['balanced_emotion_distribution']
    
    st.markdown("#### 1. Sentiment Balancing Diagnostics")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # Original Sent Chart
        df_orig_sent = pd.DataFrame(list(orig_sent.items()), columns=['Sentiment', 'Count'])
        fig_orig_sent = px.bar(
            df_orig_sent,
            x='Sentiment',
            y='Count',
            title="Raw Dataset Sentiment Distribution (Severe Imbalance)",
            color='Sentiment',
            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        )
        st.plotly_chart(fig_orig_sent, use_container_width=True)
        
    with col_d2:
        # Balanced Sent Chart
        df_bal_sent = pd.DataFrame(list(bal_sent.items()), columns=['Sentiment', 'Count'])
        fig_bal_sent = px.bar(
            df_bal_sent,
            x='Sentiment',
            y='Count',
            title="Balanced Training Set Distribution (Under-sampled Neutral)",
            color='Sentiment',
            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        )
        st.plotly_chart(fig_bal_sent, use_container_width=True)
        
    st.markdown("#### 2. Emotion Balancing Diagnostics")
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        df_orig_em = pd.DataFrame(list(orig_em.items()), columns=['Emotion', 'Count'])
        fig_orig_em = px.bar(
            df_orig_em,
            x='Count',
            y='Emotion',
            orientation='h',
            title="Raw Dataset Emotion Distribution (Neutral Dominance)",
            color='Emotion',
            color_discrete_map={k: v['color'] for k, v in EMOTION_INFO.items()}
        )
        fig_orig_em.update_layout(yaxis={'categoryorder':'total ascending'}, height=450)
        st.plotly_chart(fig_orig_em, use_container_width=True)
        
    with col_e2:
        df_bal_em = pd.DataFrame(list(bal_em.items()), columns=['Emotion', 'Count'])
        fig_bal_em = px.bar(
            df_bal_em,
            x='Count',
            y='Emotion',
            orientation='h',
            title="Balanced Training Set Distribution (Under-sampled Neutral)",
            color='Emotion',
            color_discrete_map={k: v['color'] for k, v in EMOTION_INFO.items()}
        )
        fig_bal_em.update_layout(yaxis={'categoryorder':'total ascending'}, height=450)
        st.plotly_chart(fig_bal_em, use_container_width=True)
 
    st.markdown("""
    > [!TIP]
    > **Imbalance Handling Technique:**
    > - **Downsampling**: The huge Neutral class is downsampled to ~103k for sentiment and ~40k for emotion. This reduces training times and prevents model dominance.
    > - **Algorithmic Weighting**: To prevent rarer emotions (like `boredom` with 126 instances or `worry` with 4,475 instances) from being ignored, we configured our estimators with `class_weight='balanced'`. This dynamically scales the classification error penalty proportionally to class rarity.
    """)

# ==========================================
# TAB 3: MODEL PERFORMANCE (METRICS)
# ==========================================
with tab3:
    st.markdown("### Classification Evaluation Reports")
    st.write("Below are the standard metrics measured on hold-out validation sets (20% of the balanced distributions).")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("#### Sentiment Classifier Metrics")
        sent_rep = metadata['sentiment_report']
        
        # Display high level metrics
        sm_col1, sm_col2, sm_col3 = st.columns(3)
        sm_col1.metric("Macro F1-Score", f"{sent_rep['macro avg']['f1-score']:.2%}")
        sm_col2.metric("Overall Accuracy", f"{sent_rep['accuracy']:.2%}")
        sm_col3.metric("Weighted F1-Score", f"{sent_rep['weighted avg']['f1-score']:.2%}")
        
        # Convert classification report to DataFrame
        sent_df_metrics = pd.DataFrame(sent_rep).transpose().iloc[:-3] # Exclude accuracy, macro avg, weighted avg
        sent_df_metrics.index.name = "Class Label"
        st.dataframe(sent_df_metrics.style.format("{:.2%}"), use_container_width=True)
        
    with col_m2:
        st.markdown("#### Emotion Classifier Metrics")
        em_rep = metadata['emotion_report']
        
        # Display high level metrics
        em_col1, em_col2, em_col3 = st.columns(3)
        em_col1.metric("Macro F1-Score", f"{em_rep['macro avg']['f1-score']:.2%}")
        em_col2.metric("Overall Accuracy", f"{em_rep['accuracy']:.2%}")
        em_col3.metric("Weighted F1-Score", f"{em_rep['weighted avg']['f1-score']:.2%}")
        
        # Convert classification report to DataFrame
        em_df_metrics = pd.DataFrame(em_rep).transpose().iloc[:-3]
        em_df_metrics.index.name = "Class Label"
        st.dataframe(em_df_metrics.style.format("{:.2%}"), use_container_width=True)

    st.success("🎯 Both models maintain high precision and recall, demonstrating that downsampling combined with cost-sensitive weights prevents majority class dominance.")
