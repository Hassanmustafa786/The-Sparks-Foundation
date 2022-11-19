#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading the dataset from google drive

# In[2]:


url='https://drive.google.com/file/d/1lV7is1B566UQPYzzY8R2ZmOritTW299S/view?usp=sharing'
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
df = pd.read_csv(dwn_url)


# # Exploratory Data Analysis

# In[3]:


df = pd.DataFrame(df)
df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.sample()


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


df.groupby('Region')['State'].value_counts(ascending=False)


# In[15]:


print(df['Category'].unique())
print(df['Sub-Category'].unique())


# In[16]:


print(df['Segment'].value_counts())


# # Differentiate the -ve Profit Values

# In[17]:


loss = df[df["Profit"] < 0]
loss


# In[18]:


loss.shape


# In[19]:


loss.describe()


# In[20]:


loss_total = loss["Profit"].sum()
print("The total loss is: ",loss_total)


# In[21]:


loss.groupby(loss["Segment"]).sum()


# In[22]:


loss.groupby(loss["Sub-Category"]).sum()


# In[23]:


loss["Sub-Category"].value_counts()


# # Data Visualization

# In[28]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))
ax = plt.axes()

sns.heatmap(df.isnull(),)
ax.set_facecolor("Black")


# In[32]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))
ax = plt.axes()

sns.heatmap(df.corr(),annot=True)
ax.set_facecolor("Black")


# In[48]:


sns.set(font_scale=2)
plt.figure(figsize=(30,10))
ax = plt.axes()

sort = loss["Segment"].value_counts(ascending = True)
sns.countplot(x= sort, data = loss,color="Yellow")
ax.set_facecolor("Black")


# In[40]:


sns.set(font_scale=1)
plt.figure(figsize=(30,10))
ax= plt.axes()

sns.countplot(x="State", data=loss,color="black")
ax.set_facecolor("White")


# In[41]:


sns.set(font_scale=2)
loss.hist(figsize=(30,30),color ="black")

#fig = plt.figure()
#fig.savefig("histPlot.png")
plt.show


# In[51]:


sns.set(font_scale=2)
plt.figure(figsize=(30,10))
ax=plt.axes()

plt.bar(loss["Sub-Category"],loss["Sales"],color="Purple")
plt.xlabel("Sub-Category")
plt.ylabel("Sales")
ax.set_facecolor("White")
plt.show


# In[43]:


sns.set(font_scale=2)
plt.figure(figsize=(30,10))
ax=plt.axes()

plt.bar(df["Sub-Category"],df["Sales"],color="maroon")
plt.xlabel("Sub-Category")
plt.ylabel("Sales")
ax.set_facecolor("White")
plt.show


# In[50]:


sns.set(font_scale=2)
plt.figure(figsize=(30,10))
ax=plt.axes()

plt.bar(df["Sub-Category"],df["Discount"],color="Orange")
plt.xlabel("Sub-Category")
plt.ylabel("Discount")
ax.set_facecolor("Black")
plt.show


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




