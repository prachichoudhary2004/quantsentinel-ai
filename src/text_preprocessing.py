import os
import sys
import re
import json

# Adjust path robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observability import PipelineLogger

# Compact list of common English financial and standard stopwords
STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cant', 'cannot',
    'co', 'could', 'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each',
    'few', 'for', 'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 
    'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 
    'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 
    'mustnt', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 
    'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 
    'then', 'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 
    'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent', 
    'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 
    'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
])

def clean_html_and_urls(text: str) -> str:
    """
    Strips raw HTML tags and Web URLs using regular expressions.
    """
    # Remove HTML
    text = re.sub(r'<[^>]*>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    return text

def strip_emojis(text: str) -> str:
    """
    Removes emojis and special unicode symbols.
    """
    # Keep standard alphanumeric characters, punctuation, and crypto indicators
    return re.sub(r'[^\x00-\x7F]+', ' ', text)

def crypto_tokenize_and_lemmatize(text: str) -> list:
    """
    Splits text into lowercased alphanumeric tokens, removes stopwords, 
    and applies a suffix-based light lemmatizer (to avoid NLTK downloads).
    """
    # Tokenize by splitting on non-word characters, preserving $ for prices and % for metrics
    tokens = re.findall(r'\b[a-zA-Z0-9$%\-\.]+\b', text.lower())
    
    clean_tokens = []
    for t in tokens:
        # Strip stopwords
        if t in STOPWORDS or len(t) <= 1:
            continue
            
        # Light stemmer/lemmatizer fallback
        if t.endswith('s') and not t.endswith('ss'):
            t = t[:-1]
        elif t.endswith('ed'):
            t = t[:-2]
        elif t.endswith('ing'):
            t = t[:-3]
            
        clean_tokens.append(t)
        
    return clean_tokens

def calculate_jaccard_similarity(tokens_a: list, tokens_b: list) -> float:
    """
    Computes Jaccard Similarity metric between two token sets.
    Formula: Intersection / Union
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return float(len(set_a.intersection(set_b)) / len(union))

def preprocess_news_directory():
    """
    Loads Bronze articles, applies clean-ups, filters duplicate syndications, and saves Silver Parquet/JSONs.
    """
    logger = PipelineLogger()
    logger.start_timer('text_preprocessing')
    
    bronze_dir = "data/bronze/news"
    silver_dir = "data/silver/news_clean"
    os.makedirs(silver_dir, exist_ok=True)
    
    if not os.path.exists(bronze_dir):
        print("Bronze news directory missing. Run Ingestion first.")
        return
        
    raw_files = [f for f in os.listdir(bronze_dir) if f.endswith('.json')]
    preprocessed_articles = []
    
    for f_name in raw_files:
        with open(os.path.join(bronze_dir, f_name), "r") as f:
            art = json.load(f)
            
        # 1. Cleaning raw text
        title_clean = clean_html_and_urls(strip_emojis(art['title']))
        content_clean = clean_html_and_urls(strip_emojis(art['content']))
        
        # 2. Tokenize & Lemmatize
        tokens = crypto_tokenize_and_lemmatize(content_clean)
        
        preprocessed_articles.append({
            'article_id': art['article_id'],
            'title': art['title'],
            'content': art['content'],
            'title_clean': title_clean,
            'content_clean': content_clean,
            'tokens': tokens,
            'source': art['source'],
            'author': art['author'],
            'published_date': art['published_date'],
            'url': art['url']
        })
        
    # 3. Duplicate detection using Jaccard Similarity (Syndication check)
    filtered_articles = []
    duplicate_count = 0
    
    for art in preprocessed_articles:
        is_dup = False
        for existing in filtered_articles:
            sim = calculate_jaccard_similarity(art['tokens'], existing['tokens'])
            # If 70% of tokens match, treat as syndication duplicate and drop
            if sim > 0.70:
                is_dup = True
                duplicate_count += 1
                break
        if not is_dup:
            filtered_articles.append(art)
            
    # 4. Save clean outputs
    for art in filtered_articles:
        file_path = f"{silver_dir}/{art['article_id']}.json"
        with open(file_path, "w") as f:
            json.dump(art, f, indent=4)
            
    logger.stop_timer('text_preprocessing', len(filtered_articles))
    print(f"Preprocessed finished. Cleaned: {len(filtered_articles)} files. Deduplicated: {duplicate_count} syndicated copies.")

if __name__ == "__main__":
    preprocess_news_directory()
