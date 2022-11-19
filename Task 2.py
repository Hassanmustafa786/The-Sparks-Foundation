#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

get_ipython().run_line_magic('matplotlib', 'inline')


# # loading and reading the dataset

# In[2]:


# Load the iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head() # See the first 5 rows


# In[3]:


#Checking the directory of the dataset
dir(iris)


# # EDA (Exploratory Data Analysis)

# In[4]:


df.head()


# In[5]:


df.tail()


# # Summary Statistics

# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.sample()


# # Size and Shape

# In[9]:


df.shape


# In[10]:


df.size


# In[11]:


df.memory_usage()


# # Data Types

# In[12]:


df.dtypes


# # Missing data

# In[13]:


df.isna()


# In[14]:


df.isnull().sum()


# # Duplicates

# In[15]:


df.nunique(axis=0)


# In[16]:


df.duplicated()


# # Correlation

# In[17]:


df.corr()


# # Data Visualization

# In[18]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))
ax = plt.axes()

plt.scatter(df['sepal length (cm)'],df['sepal width (cm)'],color='White')
plt.scatter(df['petal length (cm)'],df['petal width (cm)'],color='maroon')
ax.set_facecolor("black")


# In[24]:


sns.set(font_scale=1)

df.hist(figsize=(15,7),bins=25)


# # Machine Learning

# In[25]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[26]:


df.drop(['petal length (cm)', 'petal width (cm)'],axis='columns',inplace=True)


# In[27]:


km = KMeans(n_clusters=3)
km


# In[28]:


y_predicted = km.fit_predict(df[['sepal length (cm)','sepal width (cm)']])
y_predicted


# In[29]:


df['Cluster'] = y_predicted
df.head()


# In[30]:


km.cluster_centers_


# In[31]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))
ax = plt.axes()

df0 = df[df.Cluster == 0]
df1 = df[df.Cluster == 1]
df2 = df[df.Cluster == 2]

plt.scatter(df0['sepal length (cm)'], df0["sepal width (cm)"], color = 'white')
plt.scatter(df1['sepal length (cm)'], df1["sepal width (cm)"], color = 'orange')
plt.scatter(df2['sepal length (cm)'], df2["sepal width (cm)"], color = 'blue')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal width")
ax.set_facecolor("black")
plt.legend()


# In[32]:


scaler = MinMaxScaler()
scaler.fit(df[['sepal length (cm)']])
df[['sepal length (cm)']] = scaler.transform(df[['sepal length (cm)']])
df.head()


# In[33]:


scaler.fit(df[['sepal width (cm)']])
df[['sepal width (cm)']] = scaler.transform(df[['sepal width (cm)']])
df.head()


# # After scaling

# In[34]:


km= KMeans(n_clusters=3)
km


# In[35]:


y_predicted = km.fit_predict(df[['sepal length (cm)','sepal width (cm)']])
y_predicted


# In[36]:


df['Cluster'] = y_predicted
df.head()


# In[37]:


km.cluster_centers_


# In[38]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))
ax = plt.axes()

df0 = df[df.Cluster == 0]
df1 = df[df.Cluster == 1]
df2 = df[df.Cluster == 2]


plt.scatter(df0['sepal length (cm)'], df0["sepal width (cm)"], color = 'white')
plt.scatter(df1['sepal length (cm)'], df1["sepal width (cm)"], color = 'red')
plt.scatter(df2['sepal length (cm)'], df2["sepal width (cm)"], color = 'yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], color = 'purple', marker = '*', label = "Centroid",linewidths=5)


plt.xlabel("Sepal Length")
plt.ylabel("Sepal width")
ax.set_facecolor("black")
plt.legend()


# In[39]:


k_rng = range(1,11)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['sepal length (cm)','sepal width (cm)']])
    sse.append(km.inertia_)


# In[40]:


sse


# In[41]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))
ax = plt.axes()

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

plt.xlabel("K")
plt.ylabel("Sum of squared error")
plt.plot(k_rng,sse, lw=10, color='red')
plt.plot(k_rng,sse, lw=5, color='yellow')
ax.set_facecolor("White")


# In[ ]:





# In[ ]:





# In[ ]:




