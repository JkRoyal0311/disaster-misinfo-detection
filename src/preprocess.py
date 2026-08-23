import re

def clean_text(text):
    """Cleans raw tweet/news text for NLP processing."""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs/Links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions (@username)
    text = re.sub(r'\@\w+', '', text)
    # Remove hashtag symbol but keep the word
    text = re.sub(r'#', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text