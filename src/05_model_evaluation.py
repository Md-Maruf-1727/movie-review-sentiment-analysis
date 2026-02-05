#!/usr/bin/env python
# coding: utf-8

# # Model Evaluation
# 
# This step evaluates the trained Logistic Regression model on the test dataset.  
# We perform:
# - Accuracy calculation
# - Confusion matrix visualization
# - Classification report
# - Sample inspection of false positives and false negatives

# In[1]:


# Load libraries for data handling
import pandas as pd
import joblib

# For train-test split and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


# Load saved Logistic Regression model
model = joblib.load("../Models/logistic_model.joblib")

# Load saved TF-IDF vectorizer
tfidf = joblib.load("../Models/tfidf_vectorizer.joblib")

# Load saved Label Encoder
lb = joblib.load("../Models/label_encoder.joblib")


# In[3]:


# Load cleaned dataset
df = pd.read_csv("../Data/processed_data/cleaned_reviews.csv")

# Preview first 5 rows
df.head()


# ## Initial Observation
# 
# - Dataset contains `cleaned_text` and `sentiment`.
# - The model has already been trained on the training set.
# - Here we evaluate its performance on a held-out test set.

# In[4]:


# Features (text) and target (encoded sentiment)
x = df['cleaned_text']
y = lb.transform(df['sentiment'])

# Split dataset with stratification to maintain class balance
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Transform test features using the saved TF-IDF vectorizer
x_test_tfidf = tfidf.transform(xtest)


# In[5]:


# Predict sentiment on test data
y_pred = model.predict(x_test_tfidf)


# In[7]:


# Print accuracy
print(f'Accuracy Score: {accuracy_score(ytest, y_pred)}')

# Print confusion matrix
cm = confusion_matrix(ytest, y_pred)
print(f'Confusion Matrix:\n{cm}')


# # Note on Accuracy Difference
# 
# - Training phase accuracy: 88.83%  
# - Evaluation phase accuracy: 90.86%  
# 
# **Reason:** Training phase evaluated models during comparison, while evaluation phase uses stratified test set on the full dataset.  
# This difference is normal and shows good generalization of the model.

# In[8]:


# Print detailed classification report with original labels
print(f'Classification Report:\n{classification_report(ytest, y_pred, target_names=lb.classes_)}')


# In[9]:


# Identify false positives (predicted positive but actually negative)
false_pos_idx = (y_pred == 1) & (ytest == 0)
print(f'Sample False Positive Reviews:\n{xtest[false_pos_idx].head(5)}')


# In[10]:


# Identify false negatives (predicted negative but actually positive)
false_neg_idx = (y_pred == 0) & (ytest == 1)
print(f'Sample False Negative Reviews:\n{xtest[false_neg_idx].head(5)}')


# ## False Positive & False Negative Insights
# 
# - Inspecting misclassified reviews helps understand model limitations.
# - Example insights:
#   - Some negatives are predicted positive due to strong positive keywords.
#   - Some positives are predicted negative due to subtle language or negations.
# - This analysis is useful for further feature engineering or error analysis.
