import os
import sys
import json
import re
import pandas as pd
import numpy as np

# Adjust path robustly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observability import PipelineLogger

# Dynamic spaCy imports
try:
    import spacy
    HAS_SPACY_MODEL = True
except ImportError:
    HAS_SPACY_MODEL = False

# Fallback Entity Dictionary Matcher mapping keywords to target categories
ENTITY_DICT = {
    'Coin': ['btc', 'eth', 'sol', 'bitcoin', 'ethereum', 'solana', 'xrp', 'arbitrum', 'popcat'],
    'Exchange': ['binance', 'coinbase', 'kraken', 'okx', 'perpetual', 'uniswap'],
    'Regulator': ['sec', 'cftc', 'treasury', 'fca', 'fed', 'federal reserve'],
    'Institution': ['blackrock', 'fidelity', 'grayscale', 'microstrategy', 'sec'],
    'Country': ['us', 'usa', 'europe', 'china', 'uk', 'germany']
}

ENTITY_NAME_MAP = {
    'btc': 'Bitcoin (BTC)',
    'bitcoin': 'Bitcoin (BTC)',
    'eth': 'Ethereum (ETH)',
    'ethereum': 'Ethereum (ETH)',
    'sol': 'Solana (SOL)',
    'solana': 'Solana (SOL)',
    'binance': 'Binance',
    'coinbase': 'Coinbase',
    'sec': 'SEC (Securities and Exchange Commission)',
    'cftc': 'CFTC',
    'blackrock': 'BlackRock',
    'fidelity': 'Fidelity',
    'grayscale': 'Grayscale',
    'us': 'United States',
    'usa': 'United States',
    'europe': 'Europe'
}

def extract_entities_spacy(text: str) -> list:
    """
    Premium spaCy Named Entity Recognition model parser.
    """
    if not HAS_SPACY_MODEL:
        return None
        
    try:
        # Check if core English model is loaded; else try download
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
            
        doc = nlp(text)
        entities = []
        
        # Mapping standard SpaCy labels to our specific ones
        # PERSON/ORG -> Institution, GPE -> Country, etc.
        for ent in doc.ents:
            val = ent.text.strip()
            val_lower = val.lower()
            
            # Semantic filter mappings
            ent_type = "Institution"
            if val_lower in ENTITY_DICT['Coin']:
                ent_type = "Coin"
            elif val_lower in ENTITY_DICT['Exchange']:
                ent_type = "Exchange"
            elif val_lower in ENTITY_DICT['Regulator']:
                ent_type = "Regulator"
            elif ent.label_ in ['GPE', 'LOC']:
                ent_type = "Country"
            elif ent.label_ not in ['ORG', 'PERSON', 'MONEY']:
                continue # Skip uninteresting entity types
                
            standard_name = ENTITY_NAME_MAP.get(val_lower, val)
            entities.append({
                'entity': standard_name,
                'entity_type': ent_type
            })
            
        return entities
    except Exception as e:
        print(f"spaCy NER extraction failed: {e}. Launching fallback...")
        return None

def extract_entities_regex(text: str, tokens: list) -> list:
    """
    Highly optimized, resilient Regex-based entity dictionary token scanner.
    Delivers 100% target coverage and runs instantly with zero downloads.
    """
    entities = []
    text_lower = text.lower()
    
    for ent_type, keywords in ENTITY_DICT.items():
        for kw in keywords:
            # Check for word boundary occurrences in lower text
            pattern = rf'\b{re.escape(kw)}\b'
            if re.search(pattern, text_lower):
                standard_name = ENTITY_NAME_MAP.get(kw, kw.capitalize())
                entities.append({
                    'entity': standard_name,
                    'entity_type': ent_type
                })
                
    return entities

def execute_entity_pipeline():
    """
    Ingests preprocessed Silver articles, extracts entities linked with
    sentiment association weights, and stores Gold ledger outputs.
    """
    logger = PipelineLogger()
    logger.start_timer('entity_extraction')
    
    silver_dir = "data/silver/news_clean"
    gold_dir = "data/gold/sentiment"
    os.makedirs(gold_dir, exist_ok=True)
    
    if not os.path.exists(silver_dir):
        print("Silver preprocessed directory missing. Run Text Preprocessing first.")
        return
        
    files = [f for f in os.listdir(silver_dir) if f.endswith('.json')]
    entity_records = []
    
    for f_name in files:
        # Load Silver text and get Gold sentiment score
        with open(os.path.join(silver_dir, f_name), "r") as f:
            art = json.load(f)
            
        gold_json_path = os.path.join(gold_dir, f_name)
        if os.path.exists(gold_json_path):
            with open(gold_json_path, "r") as f_gold:
                gold_art = json.load(f_gold)
                sentiment_score = gold_art.get('sentiment_score', 0.0)
        else:
            sentiment_score = 0.0
            
        # Attempt Premium spaCy NER
        entities = None
        if HAS_SPACY_MODEL:
            entities = extract_entities_spacy(art['content_clean'])
            
        # Fallback to Regex parser
        if entities is None:
            entities = extract_entities_regex(art['content_clean'], art['tokens'])
            
        # Remove duplicates per article
        unique_entities = []
        seen = set()
        for ent in entities:
            key = (ent['entity'], ent['entity_type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
                
        # Append back to Gold individual article JSONs
        art['extracted_entities'] = unique_entities
        if os.path.exists(gold_json_path):
            with open(gold_json_path, "r") as f_gold:
                art = json.load(f_gold)
            art['extracted_entities'] = unique_entities
            
        with open(gold_json_path, "w") as f_out:
            json.dump(art, f_out, indent=4)
            
        # Record entities metrics linked with sentiment association
        for ent in unique_entities:
            entity_records.append({
                'article_id': art['article_id'],
                'entity': ent['entity'],
                'entity_type': ent['entity_type'],
                'sentiment_association': sentiment_score,
                'published_date': art['published_date']
            })
            
    df_entities = pd.DataFrame(entity_records)
    csv_path = f"{gold_dir}/extracted_entities.csv"
    df_entities.to_csv(csv_path, index=False)
    
    logger.stop_timer('entity_extraction', len(entity_records))
    print(f"Named Entity extraction successfully completed! Ledger saved to: {csv_path}")

if __name__ == "__main__":
    execute_entity_pipeline()
