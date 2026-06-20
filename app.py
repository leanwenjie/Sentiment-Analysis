import json
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Sentiment Analysis Workspace",
    page_icon="SA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@400;500;600;700&display=swap');

    :root {
        --font-mono: "Oxanium", sans-serif;
        --font-sans: "Oxanium", sans-serif;
        --font-serif: "Oxanium", sans-serif;
        --card: #292524;
        --ring: #F97316;
        --input: #44403C;
        --muted: #292524;
        --accent: #1e4252;
        --border: #44403C;
        --radius: 0.3rem;
        --chart-1: #F97316;
        --chart-2: #0EA5E9;
        --chart-3: #EAB308;
        --chart-4: #A8A29E;
        --chart-5: #78716C;
        --popover: #292524;
        --primary: #F97316;
        --sidebar: #292524;
        --secondary: #57534E;
        --background: #1C1917;
        --foreground: #F5F5F4;
        --destructive: #DC2626;
        --sidebar-ring: #F97316;
        --sidebar-accent: #0EA5E9;
        --sidebar-border: #44403C;
        --card-foreground: #F5F5F4;
        --sidebar-primary: #F97316;
        --muted-foreground: #A8A29E;
        --accent-foreground: #E7E5E4;
        --popover-foreground: #F5F5F4;
        --primary-foreground: #FFFFFF;
        --sidebar-foreground: #F5F5F4;
        --secondary-foreground: #E7E5E4;
        --destructive-foreground: #FFFFFF;
        --sidebar-accent-foreground: #0C2A4D;
        --sidebar-primary-foreground: #FFFFFF;
        --positive: var(--chart-3);
        --negative: var(--destructive);
        --neutral: var(--chart-4);
        --shadow-color: hsl(0 0% 5%);
        --shadow-opacity: 0.18;
        --shadow-offset-x: 0px;
        --shadow-offset-y: 2px;
        --shadow-blur: 3px;
        --shadow-spread: 0px;
        color-scheme: dark;
    }

    html, body, [class*="css"] {
        font-family: var(--font-sans);
        color: var(--foreground);
    }

    * {
        font-family: var(--font-sans) !important;
    }

    .stApp {
        background: var(--background);
        color: var(--foreground);
    }

    section[data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--sidebar-border);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .header-container {
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
        padding-bottom: 1.25rem;
    }

    .header-title {
        font-family: var(--font-sans);
        font-size: clamp(2rem, 4vw, 2.65rem);
        font-weight: 700;
        letter-spacing: 0;
        line-height: 1.05;
        margin: 0;
    }

    .header-subtitle {
        color: var(--muted-foreground);
        font-size: 1rem;
        line-height: 1.55;
        margin: 0.65rem 0 0;
        max-width: 760px;
    }

    .section-note {
        color: var(--muted-foreground);
        font-size: 0.95rem;
        line-height: 1.5;
        margin: -0.25rem 0 1rem;
    }

    .result-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow:
            var(--shadow-offset-x) var(--shadow-offset-y) var(--shadow-blur)
            var(--shadow-spread) hsl(0 0% 5% / var(--shadow-opacity));
        margin-bottom: 0.85rem;
        padding: 1rem 1.1rem;
    }

    .result-label {
        color: var(--muted-foreground);
        font-size: 0.76rem;
        font-weight: 650;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }

    .badge {
        border-radius: 4px;
        display: inline-block;
        font-size: 1rem;
        font-weight: 650;
        line-height: 1;
        padding: 0.42rem 0.62rem;
        text-transform: capitalize;
    }

    .badge-positive {
        background-color: color-mix(in srgb, var(--chart-3) 16%, transparent);
        border: 1px solid var(--chart-3);
        color: var(--chart-3);
    }

    .badge-negative {
        background-color: color-mix(in srgb, var(--destructive) 16%, transparent);
        border: 1px solid var(--destructive);
        color: var(--negative);
    }

    .badge-neutral {
        background-color: color-mix(in srgb, var(--chart-4) 16%, transparent);
        border: 1px solid var(--chart-4);
        color: var(--neutral);
    }

    .emotion-card {
        align-items: center;
        background: var(--muted);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        display: flex;
        gap: 0.75rem;
        margin-top: 0.75rem;
        padding: 0.85rem;
    }

    .emotion-marker {
        border-radius: 999px;
        flex: 0 0 auto;
        height: 0.8rem;
        width: 0.8rem;
    }

    .emotion-name {
        font-size: 1.05rem;
        font-weight: 650;
        text-transform: capitalize;
    }

    .confidence-line {
        color: var(--muted-foreground);
        font-size: 0.92rem;
        margin-top: 0.75rem;
    }

    div[data-testid="stTabs"] button {
        color: var(--foreground);
        font-weight: 600;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--primary);
    }

    h1, h2, h3 {
        font-family: var(--font-sans);
        font-weight: 700;
    }

    code, pre, kbd, div[data-testid="stDataFrame"] {
        font-family: var(--font-mono);
    }

    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow:
            var(--shadow-offset-x) var(--shadow-offset-y) var(--shadow-blur)
            var(--shadow-spread) hsl(0 0% 5% / var(--shadow-opacity));
        padding: 0.75rem;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--muted-foreground) !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--foreground) !important;
    }

    .stTextArea textarea,
    div[data-baseweb="select"] > div,
    div[data-testid="stDataFrame"] {
        background-color: var(--input) !important;
        border-color: var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--foreground) !important;
    }

    .stTextArea textarea:focus,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--ring) !important;
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--ring) 35%, transparent) !important;
    }

    .stTextArea textarea::placeholder {
        color: var(--muted-foreground) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] svg {
        color: var(--foreground) !important;
        fill: var(--foreground) !important;
    }

    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="columnheader"] {
        color: var(--foreground) !important;
    }

    .stAlert {
        background: var(--card);
        border-color: var(--border);
        color: var(--card-foreground);
    }

    .stButton button,
    button[kind="primary"] {
        background: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: var(--radius) !important;
        color: var(--primary-foreground) !important;
        font-family: var(--font-sans);
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


SENTIMENT_COLORS = {
    "positive": "#EAB308",
    "negative": "#DC2626",
    "neutral": "#A8A29E",
}

EMOTION_INFO = {
    "neutral": {"color": "#A8A29E", "sentiment": "neutral"},
    "love": {"color": "#F97316", "sentiment": "positive"},
    "happiness": {"color": "#EAB308", "sentiment": "positive"},
    "sadness": {"color": "#0EA5E9", "sentiment": "negative"},
    "relief": {"color": "#F97316", "sentiment": "positive"},
    "hate": {"color": "#DC2626", "sentiment": "negative"},
    "anger": {"color": "#B45309", "sentiment": "negative"},
    "fun": {"color": "#CA8A04", "sentiment": "positive"},
    "enthusiasm": {"color": "#F97316", "sentiment": "positive"},
    "surprise": {"color": "#0EA5E9", "sentiment": "neutral"},
    "empty": {"color": "#78716C", "sentiment": "negative"},
    "worry": {"color": "#A8A29E", "sentiment": "negative"},
    "boredom": {"color": "#78716C", "sentiment": "neutral"},
}

EXAMPLES = [
    "I seriously hate one subject to death but now i feel reluctant to drop it",
    "I was bitten by a dog, that was terrifying",
    "I am happpy when i get good results in the field of academics or athletics",
    "I fell asleep feeling angry, useless and still full of anxiety",
    "I feel like they hated me since then",
    "I feel like i ve regained another vital part of my life which is living",
    "I feel comfortable around him, it's such a relief",
    "I am sitting here typing this and wondering where i belong",
    "This was shockingly amazing! I was not expecting it at all",
]


@st.cache_resource
def load_models_and_metadata():
    """Load model pipelines and metadata, then cache them between reruns."""
    models_dir = "models"
    metadata_path = os.path.join(models_dir, "training_metadata.json")
    sent_model_path = os.path.join(models_dir, "sentiment_pipeline.joblib")
    em_model_path = os.path.join(models_dir, "emotion_pipeline.joblib")

    if not (
        os.path.exists(metadata_path)
        and os.path.exists(sent_model_path)
        and os.path.exists(em_model_path)
    ):
        return None, None, None

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    sent_pipeline = joblib.load(sent_model_path)
    em_pipeline = joblib.load(em_model_path)

    return sent_pipeline, em_pipeline, metadata


def make_bar_chart(data, x, y, title, colors, height=300, orientation="v"):
    fig = px.bar(
        data,
        x=x,
        y=y,
        orientation=orientation,
        title=title,
        color=y if orientation == "h" else x,
        color_discrete_map=colors,
        text_auto=".1%" if data[y if orientation == "v" else x].max() <= 1 else False,
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color="#F5F5F4"),
        title_font=dict(size=15),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#44403C", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


sent_pipeline, em_pipeline, metadata = load_models_and_metadata()

st.markdown(
    """
<div class="header-container">
    <h1 class="header-title">Sentiment Analysis Workspace</h1>
    <p class="header-subtitle">
        A compact Streamlit interface for testing sentence-level sentiment and emotion predictions,
        reviewing class balance, and checking validation metrics.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

if sent_pipeline is None or em_pipeline is None:
    st.error(
        "Model files were not found. Run `python train.py` first to generate the saved pipelines."
    )
    st.stop()

st.sidebar.markdown("### Project Details")
st.sidebar.write(
    "Two text classifiers are loaded from the `models` folder and evaluated against the "
    "same input sentence."
)
st.sidebar.markdown("### Training Setup")
st.sidebar.markdown(
    "- TF-IDF text features\n"
    "- Linear classifiers\n"
    "- Balanced class weights\n"
    "- Resampled training distributions"
)

tab1, tab2, tab3 = st.tabs(
    [
        "Sentence Analysis",
        "Dataset Balance",
        "Model Metrics",
    ]
)

with tab1:
    st.markdown("### Sentence Analysis")
    st.markdown(
        '<p class="section-note">Enter a sentence or start from one of the sample inputs.</p>',
        unsafe_allow_html=True,
    )

    selected_example = st.selectbox(
        "Sample input",
        ["Custom text"] + EXAMPLES,
    )
    default_text = "" if selected_example == "Custom text" else selected_example

    user_input = st.text_area(
        "Sentence",
        value=default_text,
        placeholder="Type a sentence to classify.",
        height=110,
    )

    if user_input.strip():
        sent_pred = sent_pipeline.predict([user_input])[0]
        sent_probs = sent_pipeline.predict_proba([user_input])[0]
        sent_classes = sent_pipeline.classes_

        em_pred = em_pipeline.predict([user_input])[0]
        em_probs = em_pipeline.predict_proba([user_input])[0]
        em_classes = em_pipeline.classes_

        col1, col2 = st.columns([0.9, 1.1])

        with col1:
            st.markdown("#### Results")
            sent_confidence = sent_probs[list(sent_classes).index(sent_pred)]
            em_confidence = em_probs[list(em_classes).index(em_pred)]
            em_meta = EMOTION_INFO.get(
                em_pred, {"color": "#A8A29E", "sentiment": "neutral"}
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Sentiment</div>
                    <span class="badge badge-{sent_pred}">{sent_pred}</span>
                    <div class="confidence-line">Confidence: <strong>{sent_confidence:.2%}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Emotion</div>
                    <div class="emotion-card">
                        <span class="emotion-marker" style="background: {em_meta['color']};"></span>
                        <span class="emotion-name" style="color: {em_meta['color']};">{em_pred}</span>
                    </div>
                    <div class="confidence-line">Confidence: <strong>{em_confidence:.2%}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("#### Probability Distribution")

            sent_df = pd.DataFrame(
                {
                    "Sentiment": sent_classes,
                    "Probability": sent_probs,
                }
            ).sort_values(by="Probability", ascending=True)

            fig_sent = make_bar_chart(
                sent_df,
                x="Probability",
                y="Sentiment",
                title="Sentiment probabilities",
                colors=SENTIMENT_COLORS,
                height=210,
                orientation="h",
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            em_df = pd.DataFrame(
                {
                    "Emotion": em_classes,
                    "Probability": em_probs,
                }
            ).sort_values(by="Probability", ascending=True)

            fig_em = make_bar_chart(
                em_df,
                x="Probability",
                y="Emotion",
                title="Emotion probabilities",
                colors={k: v["color"] for k, v in EMOTION_INFO.items()},
                height=360,
                orientation="h",
            )
            st.plotly_chart(fig_em, use_container_width=True)
    else:
        st.info("Enter a sentence to run the classifiers.")

with tab2:
    st.markdown("### Dataset Balance")
    st.markdown(
        """
        <p class="section-note">
            The original dataset is dominated by neutral labels. These charts compare the
            original distributions with the resampled training distributions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    orig_sent = metadata["original_sentiment_distribution"]
    bal_sent = metadata["balanced_sentiment_distribution"]
    orig_em = metadata["original_emotion_distribution"]
    bal_em = metadata["balanced_emotion_distribution"]

    st.markdown("#### Sentiment Labels")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        df_orig_sent = pd.DataFrame(list(orig_sent.items()), columns=["Sentiment", "Count"])
        fig_orig_sent = make_bar_chart(
            df_orig_sent,
            x="Sentiment",
            y="Count",
            title="Original sentiment distribution",
            colors=SENTIMENT_COLORS,
            height=330,
        )
        st.plotly_chart(fig_orig_sent, use_container_width=True)

    with col_d2:
        df_bal_sent = pd.DataFrame(list(bal_sent.items()), columns=["Sentiment", "Count"])
        fig_bal_sent = make_bar_chart(
            df_bal_sent,
            x="Sentiment",
            y="Count",
            title="Training sentiment distribution",
            colors=SENTIMENT_COLORS,
            height=330,
        )
        st.plotly_chart(fig_bal_sent, use_container_width=True)

    st.markdown("#### Emotion Labels")
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        df_orig_em = pd.DataFrame(list(orig_em.items()), columns=["Emotion", "Count"])
        fig_orig_em = make_bar_chart(
            df_orig_em,
            x="Count",
            y="Emotion",
            title="Original emotion distribution",
            colors={k: v["color"] for k, v in EMOTION_INFO.items()},
            height=450,
            orientation="h",
        )
        fig_orig_em.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_orig_em, use_container_width=True)

    with col_e2:
        df_bal_em = pd.DataFrame(list(bal_em.items()), columns=["Emotion", "Count"])
        fig_bal_em = make_bar_chart(
            df_bal_em,
            x="Count",
            y="Emotion",
            title="Training emotion distribution",
            colors={k: v["color"] for k, v in EMOTION_INFO.items()},
            height=450,
            orientation="h",
        )
        fig_bal_em.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bal_em, use_container_width=True)

    st.info(
        "The training pipeline combines resampling with balanced class weights so the "
        "model is less likely to default to the majority neutral class."
    )

with tab3:
    st.markdown("### Model Metrics")
    st.markdown(
        '<p class="section-note">Validation metrics from the saved training metadata.</p>',
        unsafe_allow_html=True,
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### Sentiment Classifier")
        sent_rep = metadata["sentiment_report"]

        sm_col1, sm_col2, sm_col3 = st.columns(3)
        sm_col1.metric("Macro F1", f"{sent_rep['macro avg']['f1-score']:.2%}")
        sm_col2.metric("Accuracy", f"{sent_rep['accuracy']:.2%}")
        sm_col3.metric("Weighted F1", f"{sent_rep['weighted avg']['f1-score']:.2%}")

        sent_df_metrics = pd.DataFrame(sent_rep).transpose().iloc[:-3]
        sent_df_metrics.index.name = "Class"
        st.dataframe(sent_df_metrics.style.format("{:.2%}"), use_container_width=True)

    with col_m2:
        st.markdown("#### Emotion Classifier")
        em_rep = metadata["emotion_report"]

        em_col1, em_col2, em_col3 = st.columns(3)
        em_col1.metric("Macro F1", f"{em_rep['macro avg']['f1-score']:.2%}")
        em_col2.metric("Accuracy", f"{em_rep['accuracy']:.2%}")
        em_col3.metric("Weighted F1", f"{em_rep['weighted avg']['f1-score']:.2%}")

        em_df_metrics = pd.DataFrame(em_rep).transpose().iloc[:-3]
        em_df_metrics.index.name = "Class"
        st.dataframe(em_df_metrics.style.format("{:.2%}"), use_container_width=True)

    st.success(
        "The saved reports indicate both classifiers are retaining useful recall across "
        "minority classes after balancing."
    )
