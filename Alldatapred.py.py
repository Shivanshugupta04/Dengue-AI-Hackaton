#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("dengue_features_train.csv", index_col = 0)
df_labels = pd.read_csv("dengue_labels_train.csv",index_col = 0)


# In[3]:


df['total_cases'] = df_labels['total_cases']


# In[4]:


df


# In[5]:


X_df = df[['year','weekofyear','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c' ,'station_precip_mm']]
y_df = df[['total_cases']]


# In[6]:


X_df.shape ,y_df.shape


# In[7]:


X_df = X_df.interpolate(method="linear")


# In[8]:


X_df.dtypes


# In[9]:


X_df.isnull().sum()


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[11]:


from sklearn.ensemble import GradientBoostingRegressor


# In[49]:


train_X, test_X, train_y, test_y = train_test_split(X_df, y_df,
                      test_size = 0.3, random_state = 75)


# In[12]:


params = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split":10,
    "learning_rate": 0.02,
    "criterion": "mse"
}


# In[96]:


reg = GradientBoostingRegressor(**params)
reg.fit(train_X, train_y)


# In[97]:


y_pred = reg.predict(train_X)


# In[98]:


mean_absolute_error(train_y, y_pred)


# In[99]:


y_pred = reg.predict(test_X)
mean_absolute_error(test_y, y_pred)


# 

# In[100]:


feats = reg.feature_importances_
names = X_df.columns


# In[79]:


plt.figure(figsize = (10, 15))
plt.barh(names, feats)


# In[56]:


mean_absolute_error(train_y, y_pred)


# In[33]:


y_pred = reg.predict(test_X)
mean_absolute_error(test_y, y_pred)


# In[13]:


reg = GradientBoostingRegressor(**params)
reg.fit(X_df, y_df)


# In[15]:


y_pred = reg.predict(X_df)
mean_absolute_error(y_df, y_pred)


# In[ ]:




