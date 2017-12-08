
# coding: utf-8

# In[1]:


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
from sklearn.metrics import accuracy_score


# In[2]:


def test_data_prep(*files):
    test_data = [pd.DataFrame(loadarff(file)[0]) for file in files]
    for i in range(len(test_data)):
        test_data[i]['class'].replace([b'0', b'1'], [0, 1], inplace = True)
    return test_data


# In[3]:



test_df = test_data_prep('data/1year.arff', 'data/2year.arff', 'data/3year.arff', 'data/4year.arff', 'data/5year.arff')
test_df


# In[4]:


def test_train_splitter(data):
    X_train_2 = []
    X_test_2 = []
    y_train_2 = []
    y_test_2 = []
    for i in range(len(data)):
        X, y = data[i].iloc[:,:-1], data[i]['class']
        X.drop(['Attr37', 'Attr21'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)
        X_train.fillna(X_train.median(), inplace=True)
        X_test.fillna(X_test.median(), inplace=True)
#         Q3 = X_train.quantile(0.75)
#         Q1 = X_train.quantile(0.25)
#         IQR = Q3 - Q1
#         min = Q1 - 1.5 * IQR
#         max = Q3 + 1.5 * IQR
#         for col in X_train.columns:
#             X_train[col][X_train[col] < min[col]] = min[col]
#             X_train[col][X_train[col] > max[col]] = max[col]
        X_train_2.append(X_train)
        X_test_2.append(X_test)
        y_train_2.append(y_train)
        y_test_2.append(y_test)
    X_train_fin = pd.concat(X_train_2)
    X_test_fin = pd.concat(X_test_2)
    y_train_fin = pd.concat(y_train_2)
    y_test_fin = pd.concat(y_test_2)
    return X_train_fin, X_test_fin, y_test_fin, y_train_fin
        
X_train, X_test, y_test, y_train = test_train_splitter(test_df)

X_train2 = X_train.reset_index(drop=True)
X_test2 = X_test.reset_index(drop=True)
y_test2 = y_test.reset_index(drop=True)
y_train2 = y_train.reset_index(drop=True)


# In[5]:


X_train2.shape, y_train2.shape, X_test2.shape, y_test2.shape


# In[6]:


y_train.value_counts()


# In[7]:


y_test.value_counts()


# In[8]:


def get_me_mah_model(X_train, y_train, X_test, y_test, Model):
    model = Model(random_state=9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100))
    print("f1_score", f1_score(y_test, y_pred))
    print("precision_score", precision_score(y_test, y_pred))
    print("recall_score", recall_score(y_test, y_pred))
    scores = [n for m, n in model.predict_proba(X_test)]
    print("roc_auc_score", roc_auc_score(y_test, scores))
    print("cm matrix", confusion_matrix(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    plt.plot(fpr, tpr)
    plt.plot([(0,0), (1,1)])
    len(y_test[y_test == 1])
    plt.show()


# In[9]:


X_train3 = X_train2.dropna()
y_train3 = y_train2.iloc[X_train3.index]
X_test3 = X_test2.dropna()
y_test3 = y_test2.iloc[X_test3.index]
smote = SMOTE(random_state=9, kind = 'borderline2')
X_smote, y_smote = smote.fit_sample(X_train3, y_train3)
sns.countplot(y_smote)
plt.show()


# In[10]:


y_train3.value_counts()


# In[11]:


y_test3.value_counts()


# In[12]:


get_me_mah_model(X_smote, y_smote, X_test3, y_test3, LogisticRegression)


# In[13]:


from sklearn.ensemble import RandomForestClassifier


# In[14]:


get_me_mah_model(X_smote, y_smote, X_test3, y_test3, RandomForestClassifier)


# In[15]:


# RandomForestClassifier(n_estimators=)


# In[16]:


model = RandomForestClassifier(random_state=9, n_estimators=100, n_jobs=-1)
model.fit(X_smote, y_smote)
y_pred = model.predict(X_test3)
print("Accuracy: %.2f%%" % (accuracy_score(y_test3, y_pred) * 100))
print("f1_score", f1_score(y_test3, y_pred))
print("precision_score", precision_score(y_test3, y_pred))
scores = [n for m, n in model.predict_proba(X_test3)]
print("recall_score", recall_score(y_test3, y_pred))
print("roc_auc_score", roc_auc_score(y_test3, scores))
print("cm matrix", confusion_matrix(y_test3, y_pred))
fpr, tpr, thresholds = roc_curve(y_test3, scores)
plt.plot(fpr, tpr)
plt.plot([(0,0), (1,1)])
len(y_test3[y_test3 == 1]),len(y_pred[y_pred == 1])
plt.show()

# In[17]:


model.feature_importances_[np.argsort(model.feature_importances_)]


# In[18]:


features_imp = pd.Series(data=model.feature_importances_, index=X_test3.columns)


# In[19]:


sorted_features = features_imp.sort_values(ascending=False)
sorted_features


# In[20]:


sorted_features.index


# In[21]:


plt.figure(figsize=(10,10))
sorted_features.plot(marker='x')
# plt.xticks(features_imp.index)
plt.show()

# In[22]:


len(sorted_features.index)


# In[23]:


# significance covered by sorted features
sorted_features[sorted_features > 0.01].sum()


# In[24]:


selected_features = sorted_features[sorted_features > 0.01]


# In[25]:


selected_features.shape


# In[26]:


X_smote2 = pd.DataFrame(data=X_smote, columns=X_train3.columns)[selected_features.index].copy()
X_test4 = X_test3[selected_features.index].copy()


# In[27]:


# train = X_smote2.copy()
# train['y'] = y_smote
# test = X_test4.copy()
# test['y'] = y_test3
# train.to_csv('train.csv')
# test.to_csv('test.csv')


# In[28]:


model = RandomForestClassifier(random_state=9, n_estimators=100, n_jobs=-1)
model.fit(X_smote2, y_smote)
y_pred = model.predict(X_test4)
print("Accuracy: %.2f%%" % (accuracy_score(y_test3, y_pred) * 100))
print("f1_score", f1_score(y_test3, y_pred))
print("precision_score", precision_score(y_test3, y_pred))
print("recall_score", recall_score(y_test3, y_pred))
scores = [n for m, n in model.predict_proba(X_test4)]
print("roc_auc_score", roc_auc_score(y_test3, scores))
print("cm matrix", confusion_matrix(y_test3, y_pred))
fpr, tpr, thresholds = roc_curve(y_test3, scores)
plt.plot(fpr, tpr)
plt.plot([(0,0), (1,1)])
len(y_test3[y_test3 == 1]), len(y_pred[y_pred == 1])
plt.show()

# In[29]:


X_smote3 = X_smote2.copy()
for col in X_smote2.columns:
    X_smote3[col] = np.log(X_smote2[col] + 1 - min(X_smote2[col]))


# In[30]:


X_test5 = X_test4.copy()
for col in X_test4.columns:
    X_test5[col] = np.log(X_test4[col] + 1 - min(X_test4[col]))


# In[31]:


X_smote2.shape


# In[32]:


model = RandomForestClassifier(random_state=9, n_estimators=100,n_jobs=-1)
model.fit(X_smote3, y_smote)
y_pred = model.predict(X_test5)
print("Accuracy: %.2f%%" % (accuracy_score(y_test3, y_pred) * 100))
print("f1_score", f1_score(y_test3, y_pred))
print("precision_score", precision_score(y_test3, y_pred))
print("recall_score", recall_score(y_test3, y_pred))
scores = [n for m, n in model.predict_proba(X_test5)]
print("roc_auc_score", roc_auc_score(y_test3, scores))
print("cm matrix", confusion_matrix(y_test3, y_pred))
fpr, tpr, thresholds = roc_curve(y_test3, scores)
plt.plot(fpr, tpr)
plt.plot([(0,0), (1,1)])
len(y_test3[y_test3 == 1]), len(y_pred[y_pred == 1])
plt.show()
