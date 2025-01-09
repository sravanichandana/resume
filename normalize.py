import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import re

# Define the normalize_text function
def normalize_text(text, use_stemming=False):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    if use_stemming:
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    else:
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back to string
    normalized_text = ' '.join(tokens)
    
    return normalized_text

# Ensure you have downloaded necessary NLTK data files
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read the Excel file
df = pd.read_excel('extracted_texts.xlsx')

# Normalize the text in the "text" column
df['normalized_text'] = df['text'].apply(lambda x: normalize_text(x, use_stemming=False))

# Save the result to a new Excel file
df.to_excel('normalized_texts.xlsx', index=False)
