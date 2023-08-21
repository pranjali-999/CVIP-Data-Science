#!/usr/bin/env python
# coding: utf-8

# ![Blue%20and%20White%20Minimalist%20Welcome%20To%20My%20Page%20Facebook%20Fundraiser%20Cover%20Photo.png](attachment:Blue%20and%20White%20Minimalist%20Welcome%20To%20My%20Page%20Facebook%20Fundraiser%20Cover%20Photo.png)

# # Objective

# In[ ]:


#Task 1 : Exploratory Data Analysis on Global Terrorism


# In[ ]:


# Perform Exploratory Data Analysis on dataset : "Global Terrorsim"
# As a Security Analyst, try to find ou the hot zone of terrorism.
# What all security issues and Insights you can derive by EDA?


# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Importing DataSet

# In[2]:


df=pd.read_csv("terrorismtask1.csv",encoding='ISO-8859-1')
print(df)


# # Performing EDA

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# # Analysing Data To Find Null Values 

# In[8]:


df.isnull()


# In[9]:


df.isnull().sum()


# In[10]:


df=df[['eventid', 'iyear', 'imonth', 'country', 'region','provstate','city','crit1', 'crit2', 'crit3','success', 'suicide', 'attacktype1','targtype1','natlty1','gname','guncertain1','claimed','weaptype1','nkill','nwound']]
df.head()


# # Remove Null Values

# In[11]:


df.isnull().any()


# In[12]:


df = df.dropna()


# # Now Check If Null Values Removed Or Not

# In[13]:


df.isnull().any()


# # All Null Values Are Now Dropped

# In[14]:


df['attacktype1'].value_counts()


# In[15]:


df['provstate'].value_counts()


# In[16]:


df['iyear'].value_counts()


# # Now Start Plotting To Analyze Dataset

# In[17]:


# Plot For Understanding Attack Types By Weapon Type

# Create a pivot table with attack type and weapon type
pivot_table = df.groupby(['attacktype1', 'weaptype1']).size().unstack(fill_value=0)

# Create the heatmap plot
plt.figure(figsize=(7, 5))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='cubehelix', linewidths=.5)
plt.xlabel('Weapon Type')
plt.ylabel('Attack Type')
plt.title('Heatmap of Attack Types by Weapon Types')
plt.tight_layout()
plt.show()


# # Plot To Understand Effected Areas

# In[18]:


plt.figure(figsize=(6, 6), dpi=100)

sns.barplot(
    y=df['provstate'].value_counts()[:15].index,
    x=df['provstate'].value_counts()[:15].values,
)

plt.title("Top 15 Areas that are Attacked")
plt.xlabel("Count")
plt.ylabel("Area")
plt.tight_layout()

plt.show()


# # Year Wise Attack Analysis

# In[19]:


plt.figure(figsize=(7, 5))
sns.kdeplot(data=df, x='iyear', color='blue')
sns.rugplot(data=df, x='iyear', height=0.1, color='blue')
plt.xlabel('Year')
plt.ylabel('Density')
plt.title('KDE Plot with Rug Plot of Terrorist Attacks by Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Attacks On Cities

# In[20]:


plt.figure(figsize=(10, 6))
sns.catplot(y='city', kind='count', data=df, order=df['city'].value_counts().nsmallest(10).index, palette='viridis')
plt.xlabel('Count')
plt.ylabel('City')
plt.title('Top 10 Least Common Cities')
plt.tight_layout()
plt.show()


# # Statewise Attack Analysis

# In[33]:


plt.figure(figsize=(6,6))
sns.barplot(x='iyear', y='eventid', data=df[df['provstate'].isin(provstate_counts.index)], hue='provstate', ci=None, palette='viridis')
plt.xlabel('Year')
plt.ylabel('Event ID')
plt.title('Bar Plot of Attacks in Top 10 States by Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title='State')
plt.show()


# # Attack analysis based on Target Type 

# In[37]:


tt = df['targtype1'].value_counts()
print(tt)
plt.figure(figsize=(7, 6))
sns.boxplot(x='targtype1', y='iyear', data=df, palette='magma')
plt.xlabel('Target Types')
plt.ylabel('Year')
plt.title('Box Plot of Target Wise Attacks by Year')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# #Conclusion:
# 
# #The frequency of terrorist attacks has increased after 2010, indicating a growing concern for security and stability globally.
# 
# #The year 2014 witnessed the highest number of terrorist attacks, highlighting a significant period of heightened activity.
# 
# #Cities like Baghdad, Karachi, Lima, and Mosul ranked highest as hotzones for terrorism, with Baghdad experiencing the highest number of attacks.
# 
# #Safer zones, such as Equator, Altai, and Burgas, seemed to experience fewer incidents of terrorism, suggesting relative stability.
# 
# #Certain cities like Boftari and Dorgali emerged as relatively safe locations with lower occurrences of terrorist attacks, possibly due to effective security measures or geographical factors.
# 
