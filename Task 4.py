#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading the dataset 

# In[2]:


df = pd.read_csv("globalterrorismdb_0718dist.csv")


# # Exploratory Data Analysis

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.shape


# In[9]:


df.size


# In[10]:


df.memory_usage()


# In[11]:


df.dtypes


# In[12]:


df.isna()


# In[13]:


df.isnull().sum()


# In[14]:


df.replace([np.nan,"NaN"],0)


# In[15]:


df = df.dropna(axis=1)
df


# In[16]:


df = df.replace("Unknown",0)
df


# In[17]:


df.describe()


# In[18]:


sns.set(font_scale=2)
plt.figure(figsize=(35,10))
ax = plt.axes()

sns.heatmap(df.isnull())
ax.set_facecolor("Black")


# In[19]:


df.columns


# In[20]:


df[['dbsource']].value_counts()


# In[21]:


df.groupby("country")["country_txt"].value_counts()


# In[22]:


df.groupby("targtype1")["targtype1_txt"].value_counts()


# In[23]:


df.groupby("weaptype1")["weaptype1_txt"].value_counts()


# In[24]:


df.groupby("attacktype1")["attacktype1_txt"].value_counts()


# In[25]:


df.groupby('iyear')['success'].value_counts()


# In[26]:


df.groupby('iyear')['suicide'].value_counts()


# In[27]:


df['success'].value_counts()


# In[28]:


df['suicide'].value_counts()


# # Data Visualization

# In[29]:



#plt.pie(x= df["country"],labels= df['country_txt'])
plt.show()


# In[30]:


sns.set(font_scale=2)
plt.figure(figsize=(30,10))
ax = plt.axes()

sort = df["region"].value_counts(ascending = True)
sns.countplot(x= sort, data = df['iyear'])
sns.color_palette("dark", as_cmap=True)
ax.set_facecolor("WHite")


# In[ ]:





# In[ ]:




