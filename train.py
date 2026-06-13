import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk

def train_and_save():
    # Ensure VADER lexicon is downloaded
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading VADER lexicon...")
        nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    print("=== Loading Dataset ===")
    dataset_path = 'emotion_sentimen_dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    df['text'] = df['text'].fillna('')
    
    # Define original sentiment mapping
    pos_emotions = {'love', 'happiness', 'relief', 'fun', 'enthusiasm'}
    neg_emotions = {'sadness', 'hate', 'anger', 'empty', 'worry'}
    
    def map_sentiment(emotion):
        if emotion in pos_emotions:
            return 'positive'
        elif emotion in neg_emotions:
            return 'negative'
        else:
            return 'neutral'
            
    df['Sentiment'] = df['Emotion'].apply(map_sentiment)
    
    # Save original distributions
    orig_sentiment_dist = df['Sentiment'].value_counts().to_dict()
    orig_emotion_dist = df['Emotion'].value_counts().to_dict()
    
    print("\nOriginal Sentiment Distribution:")
    for k, v in orig_sentiment_dist.items():
        print(f"  {k}: {v}")
        
    print("\nOriginal Emotion Distribution:")
    for k, v in orig_emotion_dist.items():
        print(f"  {k}: {v}")

    # Create directory for models
    os.makedirs('models', exist_ok=True)
    
    # ==========================================
    # 1. TRAIN SENTIMENT PIPELINE (Strict Balancing + Weak Supervision Cleaning)
    # ==========================================
    print("\n=== Preparing Sentiment Classifier ===")
    target_sent_size = 55000
    
    # Sample balanced sets
    df_neu = df[df['Sentiment'] == 'neutral'].sample(n=target_sent_size, random_state=42).copy()
    df_pos = df[df['Sentiment'] == 'positive'].sample(n=target_sent_size, random_state=42).copy()
    df_neg = df[df['Sentiment'] == 'negative'].sample(n=target_sent_size, random_state=42).copy()
    
    # Apply VADER label cleaning to the sampled neutral class (Weak Supervision)
    print("Applying VADER weak-supervision to correct neutral label noise...")
    def clean_sentiment(text):
        scores = sia.polarity_scores(str(text))
        comp = scores['compound']
        if comp <= -0.3:
            return 'negative'
        elif comp >= 0.3:
            return 'positive'
        return 'neutral'
        
    df_neu['Sentiment'] = df_neu['text'].apply(clean_sentiment)
    
    # Combine back
    df_sent_balanced = pd.concat([df_neu, df_pos, df_neg])
    balanced_sentiment_dist = df_sent_balanced['Sentiment'].value_counts().to_dict()
    print("Balanced & Cleaned Sentiment Distribution:")
    for k, v in balanced_sentiment_dist.items():
        print(f"  {k}: {v}")
        
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        df_sent_balanced['text'], df_sent_balanced['Sentiment'],
        test_size=0.2, random_state=42, stratify=df_sent_balanced['Sentiment']
    )
    
    print("Training Sentiment model (fit_intercept=False)...")
    sent_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 3), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, fit_intercept=False, class_weight='balanced', random_state=42))
    ])
    sent_pipeline.fit(X_train_s, y_train_s)
    
    y_pred_s = sent_pipeline.predict(X_test_s)
    sent_report = classification_report(y_test_s, y_pred_s, output_dict=True)
    print("Sentiment Classification Report:")
    print(classification_report(y_test_s, y_pred_s))
    
    print("Saving Sentiment Pipeline...")
    joblib.dump(sent_pipeline, 'models/sentiment_pipeline.joblib')

    # ==========================================
    # 2. TRAIN EMOTION PIPELINE (Strict Balancing + Weak Supervision Cleaning)
    # ==========================================
    print("\n=== Preparing Emotion Classifier ===")
    target_em_size = 20000
    
    # Sample balanced sets for each emotion
    em_dfs = []
    for em_val, group in df.groupby('Emotion'):
        if len(group) >= target_em_size:
            em_dfs.append(group.sample(n=target_em_size, random_state=42).copy())
        else:
            em_dfs.append(group.sample(n=target_em_size, replace=True, random_state=42).copy())
            
    # Separate the neutral emotion group to clean its label noise
    neutral_idx = next(i for i, df_group in enumerate(em_dfs) if df_group['Emotion'].iloc[0] == 'neutral')
    df_em_neutral = em_dfs[neutral_idx]
    
    print("Applying VADER weak-supervision to correct neutral emotion label noise...")
    def clean_emotion(text):
        scores = sia.polarity_scores(str(text))
        comp = scores['compound']
        if comp <= -0.3:
            return 'sadness'  # map general negative to sadness
        elif comp >= 0.3:
            return 'happiness' # map general positive to happiness
        return 'neutral'
        
    df_em_neutral['Emotion'] = df_em_neutral['text'].apply(clean_emotion)
    em_dfs[neutral_idx] = df_em_neutral
    
    df_em_balanced = pd.concat(em_dfs)
    balanced_emotion_dist = df_em_balanced['Emotion'].value_counts().to_dict()
    print("Balanced & Cleaned Emotion Distribution:")
    for k, v in list(balanced_emotion_dist.items())[:5]:
        print(f"  {k}: {v}")
    print("  ...")
    
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        df_em_balanced['text'], df_em_balanced['Emotion'],
        test_size=0.2, random_state=42, stratify=df_em_balanced['Emotion']
    )
    
    print("Training Emotion model (fit_intercept=False)...")
    emotion_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1, 3), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, fit_intercept=False, class_weight='balanced', random_state=42))
    ])
    emotion_pipeline.fit(X_train_e, y_train_e)
    
    y_pred_e = emotion_pipeline.predict(X_test_e)
    emotion_report = classification_report(y_test_e, y_pred_e, output_dict=True)
    print("Emotion Classification Report:")
    print(classification_report(y_test_e, y_pred_e))
    
    print("Saving Emotion Pipeline...")
    joblib.dump(emotion_pipeline, 'models/emotion_pipeline.joblib')

    # ==========================================
    # 3. SAVE DIAGNOSTICS & METADATA
    # ==========================================
    metadata = {
        'original_sentiment_distribution': orig_sentiment_dist,
        'balanced_sentiment_distribution': balanced_sentiment_dist,
        'original_emotion_distribution': orig_emotion_dist,
        'balanced_emotion_distribution': balanced_emotion_dist,
        'sentiment_report': sent_report,
        'emotion_report': emotion_report,
        'sentiment_classes': list(sent_pipeline.classes_),
        'emotion_classes': list(emotion_pipeline.classes_),
        'sentiment_mapping': {
            'positive': list(pos_emotions),
            'negative': list(neg_emotions),
            'neutral': ['neutral', 'boredom', 'surprise']
        }
    }
    
    with open('models/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\n=== Training Completed Successfully! ===")
    print("Saved pipeline artifacts to the 'models/' directory.")

if __name__ == '__main__':
    train_and_save()
