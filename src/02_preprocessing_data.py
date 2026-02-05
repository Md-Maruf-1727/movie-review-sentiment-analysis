#!/usr/bin/env python
# coding: utf-8

# ## Text Cleaning & Preprocessing
# 
# In this step, raw IMDB reviews are cleaned and normalized to make them suitable
# for feature extraction and machine learning models.
# 

# In[1]:


# Import required libraries
import pandas as pd
import re
import nltk

# NLP utilities from NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[2]:


# Load dataset
df = pd.read_csv("../Data/Raw/IMDB Dataset.csv")


# In[3]:


# Preview dataset
df.head()


# ## Initial Observation
# 
# - Movie reviews contain HTML tags, punctuation, numbers, and noise.
# - Raw text must be cleaned before feature extraction.

# In[ ]:


def clean_text(text):
    text = text.lower()                          # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)            # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'([!?.])\1+', r'\1', text)    # Reduce repeated punctuation
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)   # Limit repeated characters
    text = re.sub(r'\d+/\d+|\d+', '', text)      # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)          # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()     # Normalize spaces

    words = text.split()                         # Tokenize text
    stop_words = set(stopwords.words('english')) # Load stopwords
    words = [w for w in words if w not in stop_words]  # Remove stopwords

    lemmatizer = WordNetLemmatizer()              # Initialize lemmatizer
    words = [lemmatizer.lemmatize(w) for w in words]   # Lemmatize words

    return " ".join(words)


# ## Why This Cleaning Step?
# 
# - Reduces noise in raw text data
# - Standardizes words for better model learning
# - Improves overall sentiment classification performance
# 

# In[5]:


# Apply text cleaning to the review column
df['cleaned_text'] = df['review'].apply(clean_text)


# In[6]:


# Preview cleaned data
df[['review', 'cleaned_text', 'sentiment']].tail()


# In[7]:


# Save cleaned dataset for future modeling
df[['cleaned_text', 'sentiment']].to_csv(
    "../Data/processed_data/cleaned_reviews.csv",
    index=False
)

