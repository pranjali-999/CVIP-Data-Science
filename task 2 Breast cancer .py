#!/usr/bin/env python
# coding: utf-8

# ![Blue%20and%20White%20Minimalist%20Welcome%20To%20My%20Page%20Facebook%20Fundraiser%20Cover%20Photo%20%281%29.png](attachment:Blue%20and%20White%20Minimalist%20Welcome%20To%20My%20Page%20Facebook%20Fundraiser%20Cover%20Photo%20%281%29.png)

# # Golden Task- Exploratory Data Analysis on Breast Cancer Prediction

# # Importing Required Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


# # Loading Kaggle Dataset 

# In[2]:


df=pd.read_csv("Breast_Cancer.csv",encoding='ISO-8859-1')
print(df)


# # Performing EDA step by step

# In[3]:


df.head(15)


# In[4]:


df.tail(15)


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.columns


# # Finding Null Values

# In[9]:


df.isnull()


# # No null values found

# In[10]:


df.isnull().sum()


# # Droping Duplicates If Any

# In[11]:


df.drop_duplicates(inplace=True)
df.dropna(inplace=True)


# # Unique Valuves

# In[12]:


df.nunique()


# # Data Visulization

# # Age distribution

# In[22]:


plt.figure(figsize=(6,6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()


# # Differentiation Grade Distribution by Survival Status

# In[24]:


import numpy as np

plt.figure(figsize=(6, 6))
differentiation_status = df.groupby(['differentiate', 'Status']).size().unstack()
differentiation_status.plot(kind='bar', width=0.4, color=['#66b3ff', '#99ff99'])
plt.title('Differentiation Grade Distribution by Survival Status')
plt.xlabel('Differentiation Grade')
plt.ylabel('Count')
plt.xticks(np.arange(len(differentiation_status.index)), differentiation_status.index, rotation=45)
plt.legend(title='Survival Status')
plt.tight_layout()
plt.show()


# # Correlation Analysis (numerical columns)

# In[21]:


numerical_columns = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(6,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Numerical Features')
plt.show()


# # Violin Plot of Survival Months by Marital Status

# In[16]:


plt.figure(figsize=(6,6))
sns.violinplot(x='Marital Status', y='Survival Months', data=df, palette='pastel')
plt.title('Violin Plot of Survival Months by Marital Status')
plt.xticks(rotation=45)
plt.show()


# # Survival Months vs Status

# In[17]:


plt.figure(figsize=(6,6))
sns.boxplot(x='Status', y='Survival Months', data=df)
plt.title('Survival Months vs Status')
plt.show()


# # Exploratory Data Analysis: Visualizing Feature Distributions
# 
# In this section, we visualize the distributions of various features in the breast cancer prediction dataset. Each subplot showcases either the count distribution (for categorical features) or the box plot (for numerical features).
# 

# In[18]:


plt.figure(figsize=(15,16))
plotnumber = 1

for col in df.columns:
    if plotnumber <= 25:
        ax = plt.subplot(5, 5, plotnumber)
        if df[col].dtype == 'object':
            sns.countplot(x=col, data=df)
            plt.xlabel(col, fontsize=15, rotation=45)
        else:
            sns.boxplot(x=col, data=df)
            plt.xlabel(col, fontsize=15)
    plotnumber += 1

plt.tight_layout()
plt.show()


# # Tumor Size Distribution by Survival Status

# In[20]:


plt.figure(figsize=(6, 6))
sns.kdeplot(data=df, x='Tumor Size', hue='Status', fill=True, palette='Set2')
plt.title('Tumor Size Distribution by Survival Status')
plt.xlabel('Tumor Size')
plt.ylabel('Density')
plt.show()


# # Feature Importance Analysis
# 
# In this section, we delve into the significance of different features in predicting breast cancer survival. By utilizing a Random Forest classifier, we assess the importance of both categorical and numerical variables, shedding light on their respective contributions to the model's decision-making process.
# 

# In[32]:


X = df[['Age', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage',
        'differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status',
        'Progesterone Status', 'Regional Node Examined', 'Reginol Node Positive']]

categorical_columns = ['Marital Status', 'T Stage ', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']
numerical_columns = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive']

#encode categorical columns
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_columns)], remainder='passthrough')
X_encoded = preprocessor.fit_transform(X)

y = df['Status']

model = RandomForestClassifier()
model.fit(X_encoded, y)

# Get feature importances for encoded features
feature_importances = model.feature_importances_
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=categorical_columns)
all_feature_names = list(encoded_feature_names) + numerical_columns

# Plot feature importances
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=all_feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.xticks(rotation=45)
plt.show()


# # Conclusion
# This EDA uncovers pivotal survival indicators: age, tumor size, differentiation grade, and node examination. Utilizing a Random Forest classifier, we establish the importance of categorical variables (e.g., Marital Status, T Stage) and numerical variables (e.g., Age, Tumor Size) in prediction. Moving forward, we'll rigorously validate our findings with statistical tests, explore temporal trends, and employ machine learning models. Collaboration with domain experts will refine interpretations, ultimately guiding the creation of accurate predictive models for improved breast cancer prognosis.
