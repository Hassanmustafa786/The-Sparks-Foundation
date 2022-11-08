#!/usr/bin/env python
# coding: utf-8

# # Name : Hafiz Hassan Mustafa.

# # Task : Predict the percentage of the student based on the no. of study hours.
# Algorithm Used: Linear Regression.

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[93]:


#Reading the dataset
path = r'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
df = pd.read_csv(path)


# # EDA (Exploratory Data Analysis)

# # Displaying Data

# In[7]:


df.head()


# In[8]:


df.tail()


# # Summary Statistics

# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.sample()


# # Size and Shape
# 

# In[22]:


df.shape


# In[17]:


df.size


# In[19]:


df.memory_usage()


# # Data Types

# In[21]:


df.dtypes


# # Missing data

# In[23]:


df.isna()


# In[25]:


df.isnull().sum()


# # Duplicates
# 

# In[27]:


df.nunique(axis=0)


# In[29]:


df.duplicated()


# # Smallest and Largest

# In[34]:


df.nsmallest(25,['Hours'])


# In[35]:


df.nlargest(25,['Scores'])


# # Correlation

# In[36]:


df.corr()


# # Data Visualization

# In[52]:


sns.set(font_scale=1)
df.hist(figsize=(15,7),bins=25)


# In[69]:


sns.set(font_scale=2)
plt.figure(figsize=(30,10))
sns.countplot(x=df['Scores'],color='maroon')


# In[68]:


sns.set(font_scale=2)
plt.figure(figsize=(30,10))
sns.countplot(x=df['Hours'],color='black')


# # Machine Learning

# In[78]:


from sklearn import linear_model


# In[91]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))
plt.xlabel("Hours", fontsize=20)
plt.ylabel("Scores", fontsize=20)
plt.title("Linear Regression Analysis", fontsize=20)

plt.scatter(df.Hours, df['Scores'], color='red', marker='+')


# In[84]:


reg = linear_model.LinearRegression()
reg.fit(df[['Hours']], df['Scores'])


# # y = m*x + b

# In[85]:


#Predicted score if a student studies 9.25 hours/day 
#x=9.25
reg.predict([[9.25]])
#The answer is y


# In[86]:


#b
reg.intercept_


# In[87]:


#m
reg.coef_


# In[88]:


#For Confirmation
# y=m*x+b
(9.77580339*9.25)+(2.483673405373196)


# In[90]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))

plt.xlabel("Hours", fontsize=20)
plt.ylabel("Scores", fontsize=20)
plt.title("Linear Regression Analysis", fontsize=20)

plt.scatter(df.Hours, df['Scores'], color='red', marker='+')
plt.plot(df.Hours, reg.predict(df[['Hours']]), color='blue')


# In[ ]:




