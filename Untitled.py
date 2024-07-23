#!/usr/bin/env python
# coding: utf-8

# # Inspect the Data

# In[21]:


import matplotlib as plt
import pylab as pl
import numpy as np
import pandas as pd


# In[22]:


df = pd.read_csv('E:\مشاريع\qs-world-rankings-2025.csv')


# In[24]:


df.head(10)


# # Convert Data Types

# In[32]:


# Convert '2025 Rank' and '2024 Rank' to numeric, handling non-numeric values
df['2025 Rank'] = pd.to_numeric(df['2025 Rank'], errors='coerce')
df['2024 Rank'] = pd.to_numeric(df['2024 Rank'], errors='coerce')

# Convert 'QS Overall Score' to numeric
df['QS Overall Score'] = pd.to_numeric(df['QS Overall Score'], errors='coerce')


# In[33]:


df.info()


#  # Handle Missing Values

# In[34]:


# Fill missing values with the median for numeric columns
df.fillna(df.median(), inplace=True)

# For specific columns, you might want to fill with a specific value or use other strategies
df['International Faculty'].fillna(df['International Faculty'].median(), inplace=True)

# Or drop rows/columns with too many missing values
df.dropna(thresh=len(df)*0.1, axis=1, inplace=True)  # Drop columns with more than 10% missing values


# # Clean Data

# In[36]:


import re

# Function to extract numeric values from strings
def extract_numeric(value):
    match = re.search(r'\d+', str(value))
    return float(match.group()) if match else None

# Apply this function if you have ranges or mixed formats
df['Size'] = df['Size'].apply(extract_numeric)


# # Analyze Data

# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

# Summary statistics
print(df.describe())

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# # Visualize Data

# In[38]:


# Example: Distribution of 'Academic Reputation'
sns.histplot(df['Academic Reputation'])
plt.title('Distribution of Academic Reputation')
plt.xlabel('Academic Reputation')
plt.ylabel('Frequency')
plt.show()

# Example: Relationship between 'Employer Reputation' and 'QS Overall Score'
sns.scatterplot(x='Employer Reputation', y='QS Overall Score', data=df)
plt.title('Employer Reputation vs QS Overall Score')
plt.xlabel('Employer Reputation')
plt.ylabel('QS Overall Score')
plt.show()


# # Correlation Analysis

# In[39]:


import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
corr = df.corr()

# Heatmap of correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# # Distribution of Key Features

# In[40]:


sns.histplot(df['Academic Reputation'])
plt.title('Distribution of Academic Reputation')
plt.xlabel('Academic Reputation')
plt.ylabel('Frequency')
plt.show()

sns.scatterplot(x='Employer Reputation', y='QS Overall Score', data=df)
plt.title('Employer Reputation vs QS Overall Score')
plt.xlabel('Employer Reputation')
plt.ylabel('QS Overall Score')
plt.show()


# # Pairwise Relationships

# In[41]:


sns.pairplot(df[['Academic Reputation', 'Employer Reputation', 'QS Overall Score']])
plt.show()


# # Machine Learning Model

# ## Prepare Data

# In[42]:


# Select features and target variable
features = df[['Academic Reputation', 'Employer Reputation', 'Faculty Student', 'Citations per Faculty']]
target = df['QS Overall Score']


# ## Handle Missing Values

# In[43]:


# Fill missing values or drop rows/columns with NaNs
features.fillna(features.median(), inplace=True)
target.fillna(target.median(), inplace=True)


# ## Convert Categorical Feature

# In[44]:


# Example: Encoding 'Location' if it's a categorical feature
features = pd.get_dummies(features, drop_first=True)


# ## Split Data

# In[45]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# ## Choose a Model Example with Linear Regression

# In[46]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# ## Residual Analysis:
# 

# In[47]:


residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# ## Improve Model

# In[48]:


from sklearn.ensemble import RandomForestRegressor

# Initialize and train the model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predict on test data
y_pred_rf = model_rf.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Mean Squared Error: {mse_rf}')
print(f'Random Forest R^2 Score: {r2_rf}')


# In[56]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the parameter distribution
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(range(10, 50)),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Use the best model to predict
best_rf = random_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Evaluate the best model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Mean Squared Error (Tuned): {mse_rf}')
print(f'Random Forest R^2 Score (Tuned): {r2_rf}')


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the parameter distribution
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(range(10, 51)),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, verbose=2, scoring='r2', random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Use the best model to predict
best_rf = random_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Evaluate the best model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Mean Squared Error (Tuned): {mse_rf}')
print(f'Random Forest R^2 Score (Tuned): {r2_rf}')


# In[ ]:




