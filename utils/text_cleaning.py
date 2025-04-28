# utils/text_cleaning.py

import re

def preprocess_text(text):
    """
    Function to clean and preprocess the review text.
    """
    # Lowercase the text
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove special characters, numbers, punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text