import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Adjust path robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observability import PipelineLogger

# Retrieve configurable NLP execution mode from environment variables (Default: auto)
nlp_mode = os.environ.get('NLP_MODE', 'auto').lower()
print(f"Operational Topic NLP Mode configured to: {nlp_mode.upper()}")

HAS_BERTOPIC = False
if nlp_mode in ['auto', 'transformer']:
    try:
        from bertopic import BERTopic
        HAS_BERTOPIC = True
        print("Successfully imported BERTopic for topic clustering.")
    except ImportError:
        if nlp_mode == 'transformer':
            print("CRITICAL ERROR: NLP_MODE set to 'transformer' but 'bertopic' package is not installed on the system.")
            sys.exit(1)
        HAS_BERTOPIC = False
        print("BERTopic library unavailable. Falling back to Scikit-Learn LDA engine.")
else:
    # Explicit 'fallback' mode: bypass heavy libraries
    HAS_BERTOPIC = False
    print("NLP_MODE explicitly configured to 'fallback'. Bypassing BERTopic neural clustering.")

# Hardcoded keyword mapping dictionaries to identify topics semantic category
TOPIC_THEMES = {
    'ETF Approval': ['etf', 'ethereum', 'solana', 'approved', 'approval', 'filing', 'blackrock', 'fidelity'],
    'Regulation': ['regulation', 'regulators', 'sec', 'cftc', 'proposal', 'treasury', 'self-custody', 'kyc'],
    'Institutional Adoption': ['adoption', 'institutional', 'institutions', 'blackrock', 'fidelity', 'fund', 'assets'],
    'Exchange Activity': ['exchange', 'exchanges', 'withdrawals', 'outflows', 'binance', 'coinbase', 'liquidity'],
    'Macroeconomic Events': ['macro', 'macroeconomic', 'fed', 'federal', 'rates', 'funding', 'inflation', 'treasury'],
    'Security Breaches': ['security', 'breach', 'breaches', 'exploit', 'hack', 'hacked', 'bridge', 'drained'],
    'Market Liquidity': ['liquidity', 'crunch', 'outflows', 'withdrawals', 'volume', 'friction', 'liquidation']
}

def map_keywords_to_theme(top_words: list) -> str:
    """
    Scans top keywords of a topic cluster and matches it to our semantic
    financial themes based on keyword intersections.
    """
    best_theme = "Macroeconomic Events" # Default fallback
    max_matches = -1
    
    for theme, keywords in TOPIC_THEMES.items():
        matches = len(set(top_words).intersection(keywords))
        if matches > max_matches and matches > 0:
            max_matches = matches
            best_theme = theme
            
    return best_theme

def run_bertopic_modeling(docs: list) -> tuple:
    """
    Premium BERTopic topic extraction.
    """
    if not HAS_BERTOPIC:
        return None, None
        
    try:
        print("Executing unsupervised BERTopic modeling pipeline...")
        topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
        topics, probs = topic_model.fit_transform(docs)
        
        info = topic_model.get_topic_info()
        mapping = {}
        for t_id in info['Topic']:
            if t_id == -1:
                mapping[t_id] = "General / Outliers"
                continue
            words = [pair[0] for pair in topic_model.get_topic(t_id)]
            mapping[t_id] = map_keywords_to_theme(words)
            
        semantic_topics = [mapping.get(t, "Macroeconomic Events") for t in topics]
        return semantic_topics, topics
    except Exception as e:
        print(f"BERTopic execution failed: {e}.")
        if nlp_mode == 'transformer':
            print("CRITICAL: BERTopic execution failed in strict 'transformer' mode. Halting.")
            sys.exit(1)
        return None, None

def run_lda_fallback_modeling(docs: list, doc_tokens: list) -> tuple:
    """
    Highly optimized, pre-installed scikit-learn LDA topic modeling engine.
    Finds topic clusters, maps themes, and calculates frequencies at light speed.
    """
    print("Using robust Scikit-Learn LDA fallback engine for Topic modeling...")
    if len(docs) == 0:
        return [], []
        
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(docs)
    
    n_topics = min(4, len(docs))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    doc_topic_dist = lda.fit_transform(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    
    for idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-10:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        theme = map_keywords_to_theme(top_words)
        topic_keywords[idx] = theme
        print(f"Topic {idx} Top Words: {top_words[:5]} -> Semantic Mapping: {theme}")
        
    doc_topics = doc_topic_dist.argmax(axis=1)
    semantic_topics = [topic_keywords.get(t, "Macroeconomic Events") for t in doc_topics]
    
    return semantic_topics, doc_topics.tolist()

def execute_topic_modeling_pipeline():
    """
    Ingests preprocessed Silver articles, fits standard topic engines,
    and stores Gold theme arrays.
    """
    logger = PipelineLogger()
    logger.start_timer('topic_modeling')
    
    silver_dir = "data/silver/news_clean"
    gold_dir = "data/gold/sentiment"
    os.makedirs(gold_dir, exist_ok=True)
    
    if not os.path.exists(silver_dir):
        print("Silver preprocessed directory missing. Run Text Preprocessing first.")
        return
        
    files = [f for f in os.listdir(silver_dir) if f.endswith('.json')]
    docs = []
    art_meta = []
    
    for f_name in files:
        with open(os.path.join(silver_dir, f_name), "r") as f:
            art = json.load(f)
        docs.append(art['content_clean'])
        art_meta.append(art)
        
    if not docs:
        print("No preprocessed news articles found. Skipping Topic Modeling.")
        return
        
    semantic_topics, raw_topics = None, None
    if HAS_BERTOPIC:
        semantic_topics, raw_topics = run_bertopic_modeling(docs)
        
    if semantic_topics is None:
        doc_tokens = [a['tokens'] for a in art_meta]
        semantic_topics, raw_topics = run_lda_fallback_modeling(docs, doc_tokens)
        
    topic_records = []
    for idx, art in enumerate(art_meta):
        theme = semantic_topics[idx]
        cluster = raw_topics[idx]
        
        art['topic_theme'] = theme
        art['topic_cluster_id'] = int(cluster)
        
        with open(os.path.join(gold_dir, f"{art['article_id']}.json"), "w") as f_out:
            json.dump(art, f_out, indent=4)
            
        topic_records.append({
            'article_id': art['article_id'],
            'title': art['title'],
            'topic_theme': theme,
            'topic_cluster_id': int(cluster),
            'published_date': art['published_date'],
            'source': art['source']
        })
        
    df_topics = pd.DataFrame(topic_records)
    csv_path = f"{gold_dir}/topic_clusters.csv"
    df_topics.to_csv(csv_path, index=False)
    
    logger.stop_timer('topic_modeling', len(topic_records))
    print(f"Topic modeling curation successfully completed! Ledger saved to: {csv_path}")

if __name__ == "__main__":
    execute_topic_modeling_pipeline()
