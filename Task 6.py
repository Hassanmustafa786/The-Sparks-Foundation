#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading the dataset

# In[2]:


from sklearn.datasets import load_iris

iris = load_iris()

dir(iris)


# In[3]:


iris.feature_names


# # Importing the libraries and training the dataset

# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[5]:


model = DecisionTreeClassifier()


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)


# In[7]:


print(len(X_test))
print(len(X_train))


# # Fitting & Predicting

# In[8]:


model.fit(X_train,y_train)


# In[9]:


model.score(X_test,y_test)*100


# In[10]:


model.predict(iris.data[0:100])


# # Testing

# In[11]:


iris.target[23]


# In[12]:


model.predict([iris.data[23]])


# # Visualization

# In[13]:


y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)
cm


# In[14]:


plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")


# In[15]:


sns.set(font_scale=2)
plt.figure(figsize=(35,10))
ax = plt.axes()

plt.plot(y_predicted)
ax.set_facecolor("Black")
ax.grid(False)
plt.show()


# In[ ]:




