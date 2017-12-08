
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io.arff import loadarff 
from sklearn.model_selection import train_test_split


# In[2]:


def big_data_prep(*files):
    ''' 
    This function will perform the basic data preparation part by loading the raw data & converting it into a Pandas DataFrame.
    
    The *files argument is used as ther are multple files that needs to be imported. You can pass all your filenames as
    parameters separted by ','.
    
    Note: If the file is in the home directory, you can use the filenames as parameters. Else you may have to pass the full
    path along with filename
    
    This function also deals with converting the target variable for this case study in to binary(0, 1) which by default is a
    string.
    
    Post that the data from each file is split in to a test & train dataset which will maintain the Pandas DataFrame format.
    
    All the test and train datasets are concatenated to provide a final test and train data for further action.
    '''
    big_data = [pd.DataFrame(loadarff(file)[0]) for file in files]
    return pd.concat(big_data)


# In[3]:


big_data_prep('1year.arff', '2year.arff', '3year.arff', '4year.arff', '5year.arff').reset_index()


# In[4]:


def test_data_prep(*files):
    test_data = [pd.DataFrame(loadarff(file)[0]) for file in files]
    for i in range(len(test_data)):
        test_data[i]['class'].replace([b'0', b'1'], [0, 1], inplace = True)
    return test_data


# In[5]:


test_df = test_data_prep('1year.arff', '2year.arff', '3year.arff', '4year.arff', '5year.arff')
test_df


# In[6]:


def test_train_splitter(data):
    X_train_2 = []
    X_test_2 = []
    y_train_2 = []
    y_test_2 = []
    for i in range(len(data)):
        X, y = data[i].iloc[:,:-1], data[i]['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)
        X_train_2.append(X_train)
        X_test_2.append(X_test)
        y_train_2.append(y_train)
        y_test_2.append(y_test)
    X_train_fin = pd.concat(X_train_2)
    X_test_fin = pd.concat(X_test_2)
    y_train_fin = pd.concat(y_train_2)
    y_test_fin = pd.concat(y_test_2)
    return X_train_fin, X_test_fin, y_test_fin, y_train_fin
        
X_train_fin, X_test_fin, y_test_fin, y_train_fin = test_train_splitter(test_df)

y_train_fin


# In[7]:


X_train_fin_2 = X_train_fin.dropna()
y_train_fin_2 = y_train_fin.iloc[X_train_fin_2.index]


# In[8]:


from imblearn.over_sampling import SMOTE


# In[9]:


smote = SMOTE(random_state=9, kind = 'borderline2')
X_smote, y_smote = smote.fit_sample(X_train_fin_2, y_train_fin_2)
sns.countplot(y_smote)
plt.show()


# In[21]:


plt.figure(figsize=(50,50))
corr = X_train_fin_2.corr()
sns.heatmap(corr, annot=True,fmt='.1f')
plt.show()

