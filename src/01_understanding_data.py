#!/usr/bin/env python
# coding: utf-8

# # IMDB Movie Review Sentiment Analysis
# ## Data Understanding
# 
# In this notebook, we explore and understand the IMDB movie review dataset.
# The goal of this step is to gain familiarity with the data before applying any preprocessing or machine learning models.
# 

# In[2]:


# Import the pandas library for data manipulation
import pandas as pd


# In[3]:


# Load the IMDB dataset from a CSV file
df = pd.read_csv("../Data/Raw/IMDB Dataset.csv")


# In[4]:


# Display the first 5 rows of the dataset to get a quick look at the data
df.head()


# ## Initial Observation
# 
# - The dataset contains movie reviews and their corresponding sentiment labels.
# - Each row represents a single user-written movie review from IMDB.
# - The target variable `sentiment` is categorical with two classes: positive and negative.
# 

# In[5]:


# Check number of rows and columns
print(f"Row X Column: {df.shape}")


# ## Dataset Size Insight
# 
# - The dataset contains 50,000 samples.
# - This is a sufficiently large dataset for training robust machine learning models.
# - Having only two columns makes the dataset clean and easy to work with.
# 

# In[6]:


# Get detailed information about the dataset
# Includes column names, non-null counts, and data types
df.info()


# ## Data Quality Check
# 
# - No missing values are present in the dataset.
# - Both columns are stored as object types.
# - The dataset does not require any immediate cleaning.
# 

# In[13]:


# Display sample reviews to understand sentiment expressions
print(df.loc[7867]['review'])
print(df.loc[7867]['sentiment'])


# In[8]:


print(df.loc[10000]['review'])
print(df.loc[10000]['sentiment'])


# In[9]:


print(df.loc[34565]['review'])
print(df.loc[34565]['sentiment'])


# In[10]:


print(df.loc[43566]['review'])
print(df.loc[43566]['sentiment'])


# ## Why Sample Reviews Matter
# 
# - Helps understand how sentiment is expressed in natural language.
# - Reveals presence of HTML tags, sarcasm, and long-form text.
# - Guides future text preprocessing steps.

# In[12]:


# Count the number of occurrences of each sentiment (positive/negative) in the dataset
df['sentiment'].value_counts()


# ## Class Distribution Insight
# 
# - The dataset is perfectly balanced between positive and negative reviews.
# - No resampling techniques are required.
# - This balance improves model fairness and evaluation reliability.
