
# coding: utf-8

# In[1]:


# %load /home/sid/libs.txt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from scipy.io.arff import loadarff


# In[3]:


ds = loadarff('data/1year.arff')


# In[4]:


ds4 = pd.DataFrame(ds[0])


# In[5]:


ds4


# In[6]:


ds4.info()


# In[7]:


ds4.describe()


# In[8]:


ds4.info()


# In[9]:


ds5 = loadarff('data/2year.arff')


# In[10]:


ds5[0]


# In[11]:


year1data = ds4[['Attr1', 'Attr2', 'Attr3', 'Attr4', 'Attr48', 'Attr49', 'class']].copy()


# In[12]:


year1data.head()


# In[13]:


year1data['class'] = [int(n) for n in year1data['class']]


# In[14]:


year1data.head()


# In[15]:


sns.lmplot(x='Attr1', y='class', data=year1data)
plt.show()


# In[16]:


sns.lmplot(x='Attr2', y='class', data=year1data)
plt.show()


# In[17]:


sns.lmplot(x='Attr3', y='class', data=year1data)
plt.show()


# In[18]:


sns.lmplot(x='Attr4', y='class', data=year1data)
plt.show()


# In[19]:


sns.lmplot(x='Attr48', y='class', data=year1data)
plt.show()


# In[20]:


sns.lmplot(x='Attr49', y='class', data=year1data)
plt.show()


# In[21]:


year1data.corr()


# In[22]:


sns.lmplot(x='Attr1', y='Attr48', data=year1data)
plt.show()


# In[24]:


year1data['Attr1'].plot()


# In[25]:


year1data['Attr48'].plot()


# In[26]:


year1data.describe()


# In[27]:


# year1data['Attr2'].box()


# In[28]:


sns.boxplot(year1data['Attr49'])
plt.show()

# In[29]:


ds4['class'] = [int(n) for n in ds4['class']]


# In[30]:


ds4.corr().to_csv("opt.csv")


# In[31]:


sns.countplot(ds4['class']);
plt.show()

# In[33]:


ds5 = ds4.dropna().copy()
ds5.info()


# In[34]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=9, kind="borderline2")
X_smote, y_smote = smote.fit_sample(ds5[ds5.columns[:-1]], ds5['class'])
sns.countplot(y_smote)
plt.show()


# In[35]:


ds4[ds4.columns[:-1]]


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[37]:


ds4['class'] = [int(n) for n in ds4['class']]


# In[38]:


ds6 = ds4[['Attr2','Attr4','Attr7','Attr8','Attr9','Attr13','Attr33','Attr43','Attr53','class']].dropna().copy()


# In[39]:


X = ds6[ds6.columns[:-1]]
y = ds6['class']


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=9)


# In[41]:


def create_df(file_path, ds1):
   ds1 = loadarff(file_path)
   ds4 = pd.DataFrame(ds1[0])
   useful_feat = ds4[['Attr2','Attr4','Attr7','Attr8','Attr9','Attr13','Attr33','Attr43','Attr53','class']]
   return useful_feat.dropna(axis=0).copy()


# In[42]:


lr = LogisticRegression(random_state=9)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[43]:


len(y_pred[y_pred == 1])


# In[44]:


len(y_test[y_test == 1])


# In[45]:


print("f1_score", f1_score(y_test, y_pred))
print("precision_score", precision_score(y_test, y_pred))
print("recall_score", recall_score(y_test, y_pred))
print("roc_auc_score", roc_auc_score(y_test, y_pred))
print("cm matrix", confusion_matrix(y_test, y_pred))


# In[46]:


scores = [n for m, n in lr.predict_proba(X_test)]


# In[47]:


fpr, tpr, thresholds = roc_curve(y_test, scores)


# In[48]:


plt.plot(fpr, tpr)
plt.plot([(0,0), (1,1)])
plt.show()


# In[49]:


y_test


# In[50]:


len(scores)


# In[51]:


len(y_test)


# In[52]:


def big_data_prep(*files):
   big_data = [pd.DataFrame(loadarff(file)[0]) for file in files]
   return pd.concat(big_data)


# In[53]:


big_data = big_data_prep('data/1year.arff', 'data/2year.arff', 'data/3year.arff', 'data/4year.arff', 'data/5year.arff')


# In[54]:


big_data['class'] = [int(n) for n in big_data['class']]


# In[55]:


big_data[['Attr2','Attr4','Attr7','Attr8','Attr9','Attr13','Attr33','Attr43','Attr53','class']].info()


# In[56]:


big_data = big_data[['Attr2','Attr4','Attr7','Attr8','Attr9','Attr13','Attr33','Attr43','Attr53','class']].dropna().copy()


# In[57]:


X_big = big_data[big_data.columns[:-1]]
y_big = big_data['class']


# In[58]:


X_btrain, X_btest, y_btrain, y_btest = train_test_split(X_big,y_big, test_size=0.2, random_state=9)


# In[59]:


lr2 = LogisticRegression(random_state=9)
lr2.fit(X_btrain, y_btrain)
y_bpred = lr2.predict(X_btest)
bscores = [n for m, n in lr2.predict_proba(X_btest)]


# In[60]:


print("f1_score", f1_score(y_btest, y_bpred))
print("precision_score", precision_score(y_btest, y_bpred))
print("recall_score", recall_score(y_btest, y_bpred))
print("roc_auc_score", roc_auc_score(y_btest, bscores))
print("cm matrix", confusion_matrix(y_btest, y_bpred))

len(y_bpred[y_bpred == 1])
fpr, tpr, thresholds = roc_curve(y_btest, bscores)

plt.plot(fpr, tpr)
plt.plot([(0,0), (1,1)])
plt.show()