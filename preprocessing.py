"""
Advanced Text Preprocessing Module for Twitter Sentiment Analysis
Includes:
- Negation handling
- Lemmatization instead of stemming
- Emoji sentiment mapping
- Character repetition reduction
- Twitter slang normalization
"""

import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
for resource in ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ---------------------------
# Utility functions
# ---------------------------

# Emoji sentiment dictionary
emoji_dict = {
    "😍": "positive",
    "😊": "positive",
    "🙂": "positive",
    "😁": "positive",
    "😃": "positive",
    "😠": "negative",
    "😡": "negative",
    "😢": "negative",
    "😭": "negative",
    "😞": "negative",
}

# Twitter slang normalization
slang_dict = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "pls": "please",
    "plz": "please",
    "tho": "though",
    "thx": "thanks",
    "idk": "i don't know",
}

# --- URL removal ---
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)


# --- Mentions removal ---
def remove_mentions(text):
    return re.sub(r'@\w+', '', text)


# --- Hashtag symbol removal (keep the word) ---
def remove_hashtags(text):
    return text.replace("#", "")


# --- Remove everything except letters, underscores and spaces ---
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z_\s]', '', text)


# --- Convert to lowercase ---
def convert_to_lowercase(text):
    return text.lower()


# --- Normalize whitespace ---
def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

#Noemalization of slang words
def normalize_slang(text):
    """Replace common Twitter slang words."""
    words = text.split()
    normalized = [slang_dict.get(w, w) for w in words]
    return " ".join(normalized)

#Reducing character repetitions§
def reduce_repetitions(text):
    """Reduce long repeated characters: 'soooo' -> 'soo'."""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

# Emoji mapping
def map_emojis(text):
    """Convert emojis to sentiment words."""
    for emoji, label in emoji_dict.items():
        text = text.replace(emoji, f" {label} ")
    return text

# Tokenize text 
def tokenize_text(text): 
    return word_tokenize(text) 

# Remove stopwords 
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Negation handling
def handle_negations(tokens):
    """
    Convert "not good" → "not_good"
    and keep negation information.
    """
    negation_words = {"not", "no", "never", "n't"}

    result = []
    skip_next = False

    for i in range(len(tokens)):
        if skip_next:
            skip_next = False
            continue

        if tokens[i] in negation_words:
            if i + 1 < len(tokens):
                combined = tokens[i] + "_" + tokens[i + 1]
                result.append(combined)
                skip_next = True
            else:
                result.append(tokens[i])
        else:
            result.append(tokens[i])

    return result

# Lemmatization with POS tagging
def get_wordnet_pos(word):
    """Map POS tag to WordNet tag for better lemmatization."""
    tag = nltk.pos_tag([word])[0][1][0].upper()

    return {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }.get(tag, wordnet.NOUN)

# Lemmatization function
def lemmatize_words(tokens):
    """Lemmatize tokens with proper POS."""
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]


# ---------------------------
# Main cleaning pipeline
# ---------------------------

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = map_emojis(text)
    text = normalize_slang(text)
    text = reduce_repetitions(text)

    # Apply each cleaning step
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_special_characters(text)
    text = convert_to_lowercase(text)
    text = remove_extra_whitespace(text)

    # Tokenization + NLP steps
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = handle_negations(tokens)
    tokens = lemmatize_words(tokens)

    return " ".join(tokens)


# Dataset preprocessing
def preprocess_dataset(df, text_column='text'):
    print("Starting advanced preprocessing...")
    df['clean_text'] = df[text_column].apply(clean_tweet)
    print(f"Done. Processed {len(df)} tweets.")
    return df


# ---------------------------
# Test block
# ---------------------------
if __name__ == "__main__":
    test_tweets = [
        "@user I absolutely LOVE this product!!! 😍 http://example.com #amazing",
        "This is terrible. I hate it. 😠 @support pls fix",
        "Not bad, could be better tho... #meh",
        "BEST PURCHASE EVER!!! 🎉🎉🎉 https://shop.com",
        "u r soooo funny 😂😂 not good tho"
    ]

    print("\n===== DEMO =====\n")
    for t in test_tweets:
        print("Original:", t)
        print("Cleaned: ", clean_tweet(t))
        print("----------------------")
