
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
from sklearn.metrics import roc_curve, make_scorer
from scipy.io.arff import loadarff
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_importance

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
get_ipython().magic('matplotlib inline')


# In[2]:


def test_data_prep(*files):
    test_data = [pd.DataFrame(loadarff(file)[0]) for file in files]
    for i in range(len(test_data)):
        test_data[i]['class'].replace([b'0', b'1'], [0, 1], inplace = True)
    return test_data


# In[3]:



test_df = test_data_prep('data/1year.arff', 'data/2year.arff', 'data/3year.arff', 'data/4year.arff', 'data/5year.arff')
# test_df


# In[4]:


def test_train_splitter(data):
    X_train_2 = []
    X_test_2 = []
    y_train_2 = []
    y_test_2 = []
    X_val_2 = []
    y_val_2 = []
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
        X_train_v, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 9)
        X_train_2.append(X_train_v)
        X_test_2.append(X_test)
        y_train_2.append(y_train)
        y_test_2.append(y_test)
        X_val_2.append(X_val)
        y_val_2.append(y_val)
    X_train_fin = pd.concat(X_train_2)
    X_test_fin = pd.concat(X_test_2)
    y_train_fin = pd.concat(y_train_2)
    y_test_fin = pd.concat(y_test_2)
    X_val_fin = pd.concat(X_val_2)
    y_val_fin = pd.concat(y_val_2)
    return X_train_fin, X_test_fin, X_val_fin, y_test_fin, y_train_fin, y_val_fin
        
X_train, X_test, X_val, y_test, y_train, y_val  = test_train_splitter(test_df)


# In[5]:


# Defined a function for implementing hyper parameter tuning
def gridfunc(classifier, parameter, X_train, y_train):
        
    clf = classifier
    np.random.seed(9)
    parameters = parameter
    acc_scorer = make_scorer(roc_auc_score)
    
    # Run the grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, n_jobs=-1)
    grid_obj = grid_obj.fit(X_train, y_train)
    
    return grid_obj


# In[6]:


#plot mean scores for a particular grid object

def hp_cv_scores(grid_obj):
    grid_obj.cv_results_
    mean_test_scores = grid_obj.cv_results_['mean_test_score']
    mean_train_scores = grid_obj.cv_results_['mean_train_score']
    plt.figure(figsize=(10,6))
    
# #     param_values =[str(x) for x in list(grid_obj.param_grid.items())[0][1]]
#     x = np.arange(1, len(param_values)+1)
    
#     plt.plot(x,mean_train_scores, c='r', label='Train set')
#     plt.xticks(x,param_values)
#     plt.plot(x,mean_test_scores,c='g', label='Test set')
#     plt.xlabel(list(grid_obj.param_grid.items())[0][1])
#     plt.ylabel('mean scores')
#     plt.legend()
#     plt.show()


# In[7]:


classifier = RandomForestClassifier(random_state=9, n_jobs=-1)


# In[8]:


parameter = {'n_estimators': [10,50,100], 
              'max_features': ['log2', 'sqrt'], 
              'criterion': ['gini'],
              'max_depth': [5, 10], 
              'min_samples_split': [2],
              'min_samples_leaf': [1]
             }


# In[9]:


grid = gridfunc(classifier, parameter, X_val, y_val)


# In[10]:


grid.best_estimator_


# In[11]:


clf = grid.best_estimator_
    
clf.fit(X_train, y_train)


# In[39]:


clf = RandomForestClassifier(n_estimators=10, random_state=9, criterion='gini')
clf.fit(X_train, y_train)


# In[40]:


y_pred = clf.predict(X_test)


# In[41]:


print "Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100)
print "F1_score: ", f1_score(y_test, y_pred)
print "precision_score", precision_score(y_test, y_pred)
print "recall_score", recall_score(y_test, y_pred)
print "roc_auc_score", roc_auc_score(y_test, y_pred)
print "cm matrix", confusion_matrix(y_test, y_pred)
scores = [n for m, n in clf.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, scores)
plt.plot(fpr, tpr)
plt.plot([(0,0), (1,1)])
len(y_test[y_test == 1]), len(y_pred[y_pred == 1])


# In[42]:


model = XGBClassifier()
eval_set = [(X_val, y_val)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
y_pred = model.predict(X_test)
print "Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100)
print "F1_score: ", f1_score(y_test, y_pred)
print "precision_score", precision_score(y_test, y_pred)
print "recall_score", recall_score(y_test, y_pred)
print "roc_auc_score", roc_auc_score(y_test, y_pred)
print "cm matrix", confusion_matrix(y_test, y_pred)
scores = [n for m, n in model.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, scores)
plt.plot(fpr, tpr)
plt.plot([(0,0), (1,1)])
len(y_test[y_test == 1]), len(y_pred[y_pred == 1])


# In[43]:


smote = SMOTE(random_state=9, kind = 'borderline2')
X_smote, y_smote = smote.fit_sample(X_train, y_train)
X_smote2 = pd.DataFrame(data=X_smote, columns=X_train.columns)[selected_features.index].copy()
sns.countplot(y_smote)
plt.show()


# In[44]:


model = XGBClassifier()
eval_set = [(X_val, y_val)]
model.fit(X_smote, y_smote, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
y_pred = model.predict(X_test)
print "Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100)
print "F1_score: ", f1_score(y_test, y_pred)
print "precision_score", precision_score(y_test, y_pred)
print "recall_score", recall_score(y_test, y_pred)
print "roc_auc_score", roc_auc_score(y_test, y_pred)
print "cm matrix", confusion_matrix(y_test, y_pred)
scores = [n for m, n in model.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, scores)
plt.plot(fpr, tpr)
plt.plot([(0,0), (1,1)])
len(y_test[y_test == 1]), len(y_pred[y_pred == 1])

