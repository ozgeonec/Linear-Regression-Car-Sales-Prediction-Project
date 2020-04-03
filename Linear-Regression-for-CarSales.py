#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# # Loading raw data

# In[4]:


raw_data = pd.read_csv('C:\\Users\\Asus\\Desktop\\MachineLearning-Coursera\\1.04. Real-life example.csv')
raw_data.head()


# # Preprocessing

# Exploring the descriptive statistics of the variables

# In[5]:


raw_data.describe(include='all')


# Determining the variables of interest

# In[6]:


data = raw_data.drop(['Model'],axis=1)


# In[8]:


data.describe(include='all')


# Dealing with Missing Values

# In[10]:


data.isnull().sum()


# #deleting missing values

# In[12]:


data_no_mv = data.dropna(axis=0)


# In[13]:


data_no_mv.describe(include='all')


# # Exploring PDFs

# In[14]:


sns.distplot(data_no_mv['Price'])


# # Dealing with Outliers

# In[16]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')


# In[17]:


sns.distplot(data_1['Price'])


# In[18]:


sns.distplot(data_1['Mileage'])


# In[19]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[20]:


sns.distplot(data_2['Mileage'])


# In[22]:


data_3 = data_2[data_2['EngineV']<6.5]


# In[24]:


sns.distplot(data_3['EngineV'])


# In[ ]:




