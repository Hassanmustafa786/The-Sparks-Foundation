#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[1]:


import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading the dataset 

# In[2]:


df_deliveries = pd.read_csv("deliveries.csv")
df_matches = pd.read_csv("matches.csv")
df_matches.head()


# In[3]:


df_deliveries.head()


# In[4]:


print(df_deliveries.shape)
print(df_matches.shape)


# In[5]:


print(df_deliveries.isnull().sum())
print(df_matches.isnull().sum())


# In[6]:


df_deliveries.duplicated()


# In[7]:


df_matches.duplicated()


# In[8]:


df_deliveries.nunique()


# In[9]:


df_matches.nunique()


# In[10]:


winner_list = df_matches.groupby("city")["winner"].value_counts()
print(winner_list.head(20))
print(winner_list.shape)


# In[11]:


sns.set(font_scale=2)
plt.figure(figsize=(35,10))
ax = plt.axes()

sns.heatmap(df_deliveries.isnull())
ax.set_facecolor("Black")


# In[12]:


sns.set(font_scale=2)
plt.figure(figsize=(35,10))
ax = plt.axes()

sns.heatmap(df_matches.isnull())
ax.set_facecolor("Black")


# In[13]:


print(df_deliveries.columns)
print(df_matches.columns)


# In[14]:


sns.set(font_scale=2)
df_matches.hist(figsize=(30,30))
plt.show


# In[15]:


sns.set(font_scale=2)
df_deliveries.hist(figsize=(30,30))
plt.show


# # This Profile Report visualize everything. You can filter out all the details that you need.

# In[3]:


profile_matches = ProfileReport(df_matches, title="Pandas Profiling Report")
profile_matches


# In[4]:


profile_deliveries = ProfileReport(df_deliveries, title="Pandas Profiling Report")
profile_deliveries


# In[18]:


print(df_deliveries.columns)
print(df_matches.columns)


# In[19]:


print(df_deliveries['batsman'].nunique())
all_batsman = df_deliveries['batsman'].unique()
all_batsman


# # Top 10 Batsmen

# In[20]:


batting_tot=df_deliveries.groupby('batsman').apply(lambda x:np.sum(x['batsman_runs'])).reset_index(name='Runs')
batting_sorted=batting_tot.sort_values(by='Runs',ascending=False)
top_batsmen=batting_sorted[:10] 
print('The Top 10 Batsmen in thr Tournament are:\n',top_batsmen)
fig = px.bar(top_batsmen, x='batsman', y='Runs',
             hover_data=['batsman'], color='Runs',title='Top 10 Batsmen in IPL- Seasons 2008-2019')
fig.show()


# # Matches played at the city

# In[21]:


matches_played = df_matches.groupby('city')[['city']].count()
print(matches_played.shape)
matches_played


# # Matches played at the venue

# In[22]:


venue_played = df_matches.groupby('venue')[['venue']].count()
print(venue_played.shape)
venue_played


# In[23]:


df_deliveries.columns


# In[28]:


df_deliveries.groupby('fielder')[['dismissal_kind']].count()


# In[30]:


df_deliveries.groupby('batsman')[['batsman', 'wide_runs','bye_runs',
                                  'legbye_runs', 'noball_runs', 'penalty_runs',
                                  'batsman_runs', 'extra_runs', 'total_runs']].sum()


# In[ ]:





# In[ ]:





# In[ ]:




