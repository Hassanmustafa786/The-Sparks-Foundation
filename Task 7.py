#!/usr/bin/env python
# coding: utf-8

# # Task : 7 (STOCK MARKET PREDICTION)

# In this I have predicted if a companies stock will increase or decrease based on news headlines using sentiment analysis.
# 
# This model will determine if the price of a stock will increase or decrease based on the sentiment of top news article headlines for the current day using Python and machine learning.
# 
# I have used both numerical and textual data for this.
# 
# (i) Time series analysis is performed on the Stock data.
# 
# (ii) Sentiment analysis is performed on the News data.
# 
# (iii) An analysis is performed by merging both the data to predict if the Close price of the stock will increase or decrease.

# # Importing the libraries

# In[1]:


get_ipython().system('pip install pandas-datareader')
get_ipython().system('pip install pmdarima')
get_ipython().system('pip install textBlob')
get_ipython().system('pip install vaderSentiment')


# In[2]:


import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


# # Reading the dataset

# In[3]:


start = datetime.datetime(2020, 1, 1)
end = datetime.datetime.today()


# In[4]:


stocks = web.DataReader("AAPL", 'yahoo', start, end)


# # EDA

# In[5]:


stocks.info()


# In[6]:


stocks.describe()


# In[7]:


stocks.reset_index(inplace=True)


# In[8]:


stocks.head()


# In[9]:


stocks.tail()


# In[10]:


stocks.shape


# # Pandas Profiling

# In[54]:


from pandas_profiling import ProfileReport

profile = ProfileReport(stocks, title="Pandas Profiling Report")
profile


# # Data Cleaning & Missing Values

# In[11]:


stocks['Date'] = pd.to_datetime(stocks['Date'])
stocks.head()


# In[12]:


stocks.isnull().sum()


# # Data Visualization

# In[13]:


sns.set(font_scale=2)
plt.figure(figsize=(35,10))

stocks.hist(figsize=(30,30),grid= False,color = 'Red')

plt.show


# In[14]:


sns.set(font_scale=1)
plt.figure(figsize=(10,5))
ax = plt.axes()

sns.heatmap(stocks.corr(),annot=True)
ax.set_facecolor("Black")


# In[15]:


sns.set(font_scale=2)
plt.figure(figsize=(35,10))
ax = plt.axes()

plt.title('High & Low Price of Stocks')
plt.xlabel('Days')
plt.ylabel('High & Low')
plt.plot(stocks['High'],color = "Orange")
plt.plot(stocks['Low'],color = "White")

ax.set_facecolor("Black")
ax.grid(False)
plt.show()


# In[16]:


sns.set(font_scale=2)
plt.figure(figsize=(35,10))
ax = plt.axes()

plt.title('Closing Price of Stocks')
plt.xlabel('Days')
plt.ylabel('Close')
plt.plot(stocks['Close'],color = "Red")

ax.set_facecolor("Black")
ax.grid(False)
plt.show()


# In[17]:


sns.set(font_scale=2)
plt.figure(figsize=(35,10))
ax = plt.axes()

plt.plot(stocks['Open'],color = 'Black')
plt.xlabel('Days')
plt.ylabel('Open Price')
plt.title('Opening price of Stocks')

ax.set_facecolor("Orange")
ax.grid(False)
plt.show()


# In[18]:


close = stocks['Close']
returns = close / close.shift(1) - 1

sns.set(font_scale=2)
plt.figure(figsize=(35,10))
ax = plt.axes()

returns.plot(label='Return', color = 'Black')
plt.title("Stock Returns")

ax.set_facecolor("White")
ax.grid(False)
plt.show()


# # Time series Analysis

# # Splitting the data

# In[19]:


train = stocks[:500]
test = stocks[500:]


# In[20]:


train.shape, test.shape


# # Analyzing news dataset

# In[21]:


news=pd.read_csv("india-news-headlines.csv")
news.head()


# In[22]:


news['publish_date'] = pd.to_datetime(news['publish_date'],format= '%Y%m%d')
news.head()


# In[23]:


news.isna().sum()


# In[24]:


news.shape


# In[25]:


(news.columns)


# In[26]:


#Head Categories with cities
news['headline_category'].value_counts().head()


# In[27]:


cities = news[news['headline_category'].str.contains('^city\.[a-z]+$', regex=True)]
cities.head(10)


# In[28]:


city = pd.DataFrame(columns = ['city_name'])
city['city_name'] = cities.headline_category.str.split('.',expand = True)[1]
cities = pd.concat([cities, city], axis = 1)
cities.head()


# In[29]:


cities.drop('headline_category', inplace =True,axis =1)


# In[30]:


cities.head()


# In[31]:


cities_grouped = cities.groupby(cities['city_name']).agg({'headline_text':'count'})
cities_grouped.head()


# In[32]:


cities_grouped.rename(columns = {'headline_text':'headline_count'}, inplace = True)


# In[33]:


cities_grouped = cities_grouped.sort_values(by='headline_count',ascending=False)
cities_grouped.head()


# In[34]:


top10cites = cities_grouped.head(10)

def figure_plot(top10cites,title1):
    fig = px.line(top10cites,title =title1)
    for i in top10cites.columns[0:]:
        fig.add_bar(x= top10cites.index ,y = top10cites['headline_count'],name = i)
    fig.show()


# In[35]:


import plotly.express as px

figure_plot(top10cites,'Count of Headlines for top10 Cities')


# In[36]:


cities.head()


# In[37]:


#Analaysing "HEADLINE_CATEGORY" with "CATEGORIES"
news.head()


# In[38]:


news['category']=news['headline_category'].str.split('.').map(lambda x : x[0])


# In[39]:


categories = news.groupby(['category']).agg({'headline_text':'count'}).sort_values(by='headline_text',ascending = False)
news_cat=categories.head(10)
news_cat.reset_index(inplace = True)
news_cat


# In[40]:


sns.set(font_scale=2)
plt.figure(figsize=(35,10))
ax = plt.axes()

plt.bar(news_cat.category,height= news_cat.headline_text, color = 'Black')
plt.xlabel('Category')
plt.ylabel('Number of articles')
plt.title('Top 10 Categories')

ax.set_facecolor("White")
ax.grid(False)
plt.show()


# In[41]:


news.drop('headline_category', inplace  = True, axis =1)
news.head()


# # Sentimental Analaysis (Assigning Subjectivity & Polarity to the Headlines)

# In[44]:


from textblob import TextBlob


# In[45]:


# Create a function to get the subjectivity
def Subjectivity(text):
       return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def Polarity(text):
      return  TextBlob(text).sentiment.polarity


# In[47]:


news['Subjectivity'] =news['headline_text'].apply(Subjectivity)
news['Polarity'] =news['headline_text'].apply(Polarity)


# In[ ]:


import nltk
nltk.download()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

senti = SentimentIntensityAnalyzer()


# In[ ]:


news['Compound'] = [senti.polarity_scores(s)['compound'] for s in news['headline_text']]
news['Negative'] = [senti.polarity_scores(s)['neg'] for s in news['headline_text']]
news['Neutral'] = [senti.polarity_scores(s)['neu'] for s in news['headline_text']]
news['Positive'] = [senti.polarity_scores(s)['pos'] for s in news['headline_text']]


# In[ ]:


news.head()


# # Hybrid model - Combining Stocks data and news data

# In[ ]:


news.rename(columns = {'publish_date':'Date'}, inplace = True)


# In[ ]:


df_merge = pd.merge(stocks, news, how='inner', on=['Date'])
df_merge.head()


# In[ ]:


df = df_merge[['Close','Subjectivity', 'Polarity', 'Compound', 'Negative', 'Neutral' ,'Positive']]
df.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
new_df = pd.DataFrame(sc.fit_transform(df))
new_df.columns = df.columns
new_df.index = df.index
new_df.head()


# # Splitting Data

# In[ ]:


X = new_df.drop('Close', axis=1)
y =new_df['Close']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state = 0)


# In[ ]:


X_train.shape , X_test.shape


# In[ ]:


def func_graph(results,names):
    fig = plt.figure()
    fig.suptitle('MSE value of all Algorithms Comparison')
    ax = fig.add_subplot(111)
    width = 0.5        
    bars=plt.bar(names,results, width, align='center')
    ax.set_xticklabels(names)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval +0.005, yval)
    plt.show()


# In[ ]:


from sklearn import metrics

def metric_calc(name,model,category, X_train, Y_train, X_test, Y_test):
    if category =='TRAINING DATA' :
        X_data= X_train
        Y_data=Y_train
    else :
        X_data= X_test
        Y_data=Y_test
        
    model.fit(X_train, Y_train)
    predictions = model.predict(X_data)
    mse =round(metrics.mean_squared_error(predictions,Y_data),4)   
    print('For ', name, 'MSE-Value is ', mse)
    return mse


# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import xgboost 
import lightgbm


# In[ ]:


def func_modelling(i) :
    count=0
    count=count+1
    X = X_train[i]
    Y = Y_train
    x_test = X_test[i]
    seed = 7
    # preparing models list
    models = []
    models.append(('Decision Tree',' DecisiontreeRegressor  ', DecisionTreeRegressor()))
    models.append(('Random Forest',' RandomForestRegressor  ', RandomForestRegressor()))
    models.append(('XG Boost',' XGBRegressor  ', xgboost.XGBRegressor()))
    models.append(('LG Boost',' LGBMRegressor ', lightgbm.LGBMRegressor()))
    models.append(('ADA Boost',' AdaBoostRegressor ', AdaBoostRegressor()))
    results_train = []
    results_test = []
    names = []
    scoring = 'MSE'

    print('Metrics calcuated while TRANING the model')
    for name,label, model in models:
            cv_results_train=metric_calc(name,model,'TRAINING DATA',X,Y, x_test,Y_test)
            results_train.append(cv_results_train)
            names.append(name)
    func_graph(results_train,names)
    
    print('Evaluating the model on TESTING DATA')
    for name,label, model in models:
            cv_results_test=metric_calc(name,model,'TESTING DATA',X,Y, x_test,Y_test)
            results_test.append(cv_results_test)
            #names.append(name)
    func_graph(results_test,names)


# # Training the model

# In[ ]:


func_modelling(X_train.columns)


# # Pandas Profiling

# In[55]:


profile_df = ProfileReport(new_df, title="Pandas Profiling Report")
profile_df


# In[ ]:




