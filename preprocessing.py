import re
import spacy
from stop_words import get_stop_words

# Define function to prepare input sentence for tfidf vectorizer
def preprocess_text(text):

    # Loads in stop words to speed up the proccessing
    stop_words = get_stop_words("pl")

    # Load text normalizer
    normalizer = spacy.load("pl_core_news_sm")

    # Delete special charakters like @!!$%
    cleaned = re.sub(r"[^\w\s]", "", text).lower()

    # Lemmatize and tokenize text
    normalized = normalizer(cleaned)
    normalized = [token.lemma_ for token in normalized if token.text != "\n"]

    # Delete stopwords
    normalized_cleaned = " ".join([i for i in normalized if i not in stop_words])
    return normalized_cleaned
