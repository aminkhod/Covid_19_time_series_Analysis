#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import datetime
from tqdm import tqdm
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import LinearSVR
import pickle
from sklearn.metrics import mean_squared_error 
warnings.filterwarnings("ignore")


# In[65]:


data = pd.read_excel('covid_19_data.xlsx',sheet_name='per day')
data.head()


# In[83]:


X = np.reshape(list(data['days']),(-1, 1))
X_train = X[:int(.9*len(X))]
y = np.reshape(list(data['Sum of Deaths']),(-1, 1))
y_train = y[:int(.9*len(y))]
X_valid = X[int(.9*len(X)):]
y_valid = y[int(.9*len(y)):]


# In[84]:


# define eval metrics
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = y.get_label()
    return "rmspe", rmspe(y,yhat)


# ## Lasso Regression

# In[85]:


lasso_regression = Lasso()
parameters = {'alpha':[10**-5, 10**-4, 10**-3, 10**-2, 0.1, 1, 10, 10**2, 10**3, 10**4, 10**5]}
tscv = TimeSeriesSplit(n_splits=3)
clf = GridSearchCV(lasso_regression, parameters, cv=tscv, scoring='neg_mean_squared_error', verbose = 2, return_train_score=True, n_jobs = -1)
clf.fit(X_train, np.log2(y_train))

results = pd.DataFrame.from_dict(clf.cv_results_)


# In[86]:


train_score= results['mean_train_score']
train_score_std= results['std_train_score']
cv_score = results['mean_test_score'] 
cv_score_std= results['std_test_score']
param_alpha = results['param_alpha']
results.head(5)


# In[87]:


plt.plot(np.log10(param_alpha.astype(float)), np.abs(train_score), label = 'Train')
plt.plot(np.log10(param_alpha.astype(float)), np.abs(cv_score), label = 'Validation')
plt.xlabel('Log Alpha')
plt.ylabel('MSE')
plt.legend()
plt.show()


# In[102]:


lasso_regression = Lasso(alpha = 10)
lasso_regression.fit(X_train, np.log1p(y_train))

y_train_pred = lasso_regression.predict(X_train)  
y_valid_pred = lasso_regression.predict(X_valid)
print(rmspe(y_train, np.expm1(y_train_pred)))
print(rmspe(y_valid, np.expm1(y_valid_pred)))


# In[103]:


print('MSE of train: ',min(np.abs(train_score)))
print('MSE of test: ',min(np.abs(cv_score)))


# In[109]:


plt.plot(X_train, y_train, label = 'Train')
plt.plot(X_valid, np.power(2.5,y_valid_pred), label = 'Prediction')

plt.legend()
plt.show()


# ## Ridge Regression

# In[110]:


ridge_regression = Ridge()
parameters = {'alpha':[10**-5, 10**-4, 10**-3, 10**-2, 0.1, 1, 10, 10**2, 10**3, 10**4, 10**5]}
tscv = TimeSeriesSplit(n_splits=3)
clf = GridSearchCV(ridge_regression, parameters, cv=tscv, scoring='neg_mean_squared_error', verbose = 2, return_train_score=True, n_jobs = -1)
clf.fit(X_train, np.log1p(y_train))

results = pd.DataFrame.from_dict(clf.cv_results_)


# In[111]:


train_score= results['mean_train_score']
train_score_std= results['std_train_score']
cv_score = results['mean_test_score'] 
cv_score_std= results['std_test_score']
param_alpha = results['param_alpha']
results.head(5)


# In[112]:


plt.plot(np.log10(param_alpha.astype(float)), np.abs(train_score), label = 'Train')
plt.plot(np.log10(param_alpha.astype(float)), np.abs(cv_score), label = 'Validation')
plt.xlabel('Log Alpha')
plt.ylabel('MSE')
plt.legend()
plt.show()


# In[113]:


print('MSE of train: ',min(np.abs(train_score)))
print('MSE of test: ',min(np.abs(cv_score)))


# In[114]:


ridge_regression = Ridge(alpha = 1000)
ridge_regression.fit(X_train, np.log1p(y_train))

y_train_pred = ridge_regression.predict(X_train)  
y_valid_pred = ridge_regression.predict(X_valid)
print(rmspe(y_train, np.expm1(y_train_pred)))
print(rmspe(y_valid, np.expm1(y_valid_pred)))


# In[ ]:


plt.plot(X_train, y_train, label = 'Train')
plt.plot(X_valid, np.power(2.5,y_valid_pred), label = 'Prediction')

plt.legend()
plt.show()


# ## Linear SVR

# In[115]:


svr_regression = LinearSVR()
parameters = {'C':[10**-4, 10**-3, 10**-2, 0.1, 1, 10, 10**2, 10**3, 10**4]}
tscv = TimeSeriesSplit(n_splits=3)
clf = GridSearchCV(svr_regression, parameters, cv=tscv, scoring='neg_mean_squared_error', verbose = 2, return_train_score=True, n_jobs = -1)
clf.fit(X_train, np.log1p(y_train))

results = pd.DataFrame.from_dict(clf.cv_results_)


# In[116]:


train_score= results['mean_train_score']
train_score_std= results['std_train_score']
cv_score = results['mean_test_score'] 
cv_score_std= results['std_test_score']
param_C = results['param_C']
results.head(5)


# In[117]:


plt.plot(np.log10(param_C.astype(float)), np.abs(train_score), label = 'Train')
plt.plot(np.log10(param_C.astype(float)), np.abs(cv_score), label = 'Validation')
plt.xlabel('Log C')
plt.ylabel('MSE')
plt.legend()
plt.show()


# In[118]:


print('MSE of train: ',min(np.abs(train_score)))
print('MSE of test: ',min(np.abs(cv_score)))


# In[119]:


svr_regression = LinearSVR(C = 1)
svr_regression.fit(X_train, np.log1p(y_train))

y_train_pred = svr_regression.predict(X_train)  
y_valid_pred = svr_regression.predict(X_valid)
print(rmspe(y_train, np.expm1(y_train_pred)))
print(rmspe(y_valid, np.expm1(y_valid_pred)))


# In[ ]:


plt.plot(X_train, y_train, label = 'Train')
plt.plot(X_valid, np.power(2.5,y_valid_pred), label = 'Prediction')

plt.legend()
plt.show()


# ## Decision Tree

# In[120]:


decision_tree = DecisionTreeRegressor()

tscv = TimeSeriesSplit(n_splits=3)
parameters = {'min_samples_split':[100, 200, 500], 'max_depth':[10, 20, 30, 50]}
clf = GridSearchCV(decision_tree, parameters, cv=tscv, scoring='neg_mean_squared_error', verbose = 2, return_train_score=True, n_jobs = -1)
clf.fit(X_train, np.log1p(y_train))

results = pd.DataFrame.from_dict(clf.cv_results_)


# In[121]:


clf.best_params_


# In[122]:


train_auc= results['mean_train_score']
train_auc_std= results['std_train_score']
cv_auc = results['mean_test_score'] 
cv_auc_std= results['std_test_score']
param_max_depth = results['param_max_depth']
param_min_samples_split = results['param_min_samples_split']
results.head(8)


# In[123]:


data_train = results.pivot('param_min_samples_split', 'param_max_depth', 'mean_train_score')
sns.heatmap(data_train, annot=True)


# In[124]:


data_train = results.pivot('param_min_samples_split', 'param_max_depth', 'mean_test_score')
sns.heatmap(data_train, annot=True)


# In[131]:


best_max_depth = 20
best_min_samples_split = 100

decision_tree = DecisionTreeRegressor(max_depth= best_max_depth, min_samples_split=best_min_samples_split)
decision_tree.fit(X_train, np.log1p(y_train))

y_train_pred = decision_tree.predict(X_train)  
y_valid_pred = decision_tree.predict(X_valid)


# In[132]:


print('MSE of train: ',mean_squared_error(np.log1p(y_train), y_train_pred))
print('MSE of test: ',mean_squared_error(np.log1p(y_valid), y_valid_pred))


# In[133]:


print(rmspe(y_train, np.exp(y_train_pred)))
print(rmspe(y_valid, np.exp(y_valid_pred)))


# In[135]:


plt.plot(X_train, y_train, label = 'Train')
plt.plot(X_valid, np.power(2.75,y_valid_pred), label = 'Prediction')

plt.legend()
plt.show()


# In[ ]:




