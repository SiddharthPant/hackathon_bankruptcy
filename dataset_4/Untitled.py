
# coding: utf-8

# In[2]:


# %load /home/sid/libs.txt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.io.arff import loadarff
from sklearn.ensemble import RandomForestClassifier
get_ipython().magic('matplotlib inline')


# In[3]:


def big_data_prep(*files):
   big_data = [pd.DataFrame(loadarff(file)[0]) for file in files]
   return pd.concat(big_data)


# In[4]:


big_data = big_data_prep('data/1year.arff', 'data/2year.arff', 'data/3year.arff', 'data/4year.arff', 'data/5year.arff')


# In[5]:


big_data['class'] = [int(n) for n in big_data['class']]


# In[15]:


big_data.isnull().sum().sort_values(ascending=False)


# In[17]:


big_data.copy().dropna(how='all').shape


# In[18]:


big_data.shape


# In[35]:


sns.boxplot(big_data['Attr27'])


# In[38]:


big_data['Attr27'].value_counts().sort_values()


# In[58]:


test_data = big_data[['Attr27','class']]


# In[59]:


data27 = test_data[test_data['class'] == 1]['Attr27']


# In[60]:


sns.boxplot(data27)


# In[61]:


data27.describe()


# In[62]:


data27_2 = data27.dropna().copy()


# In[63]:


sns.distplot(data27_2)


# In[66]:


data27.fillna(value=0.83, inplace=True)


# In[67]:


sns.distplot(data27)


# In[65]:


data27.isnull()


# In[69]:


big_data['Attr27'].median()


# In[88]:


Q3 = big_data.quantile(0.75)
Q1 = big_data.quantile(0.25)

IQR = Q3 - Q1

min = Q1 - 1.5 * IQR

max = Q3 + 1.5 * IQR


# In[85]:


sns.distplot(filtered['Attr27'])


# In[87]:


filtered[filtered['class'] == 1].shape


# In[104]:


for col in big_data.columns:
    big_data[col][big_data[col] < min[col]] = min[col]
    big_data[col][big_data[col] > max[col]] = max[col]


# In[107]:


dsns.distplot(big_data.dropna())


# In[101]:


big_data['Attr27'][big_data['Attr27'] < min['Attr27']] = min['Attr27']
big_data['Attr27'][big_data['Attr27'] > max['Attr27']] = max['Attr27']


# In[103]:


sns.distplot(big_data['Attr27'].dropna())


# In[108]:


sns.boxplot(big_data['Attr27'].dropna())


# In[109]:


big_data['Attr27'].median()

