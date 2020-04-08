#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# # Loading Raw Data

# In[12]:


raw_data = pd.read_csv('C:\\Users\\Asus\\Desktop\\MachineLearning-Coursera\\LinearRegrssion-CarSales Project\\1.04. Real-life example.csv')
raw_data.head()


# # Preprocessing

# In[13]:


raw_data.describe(include='all')


# In[14]:


data = raw_data.drop(['Model'],axis=1)


# In[15]:


data.describe(include='all')


# ### Dealing with missing values

# In[16]:


data_no_mv = data.dropna(axis=0)


# In[17]:


data_no_mv.describe(include='all')


# # Exploring pdfs

# In[18]:


sns.distplot(data_no_mv['Price'])


# # Dealing with Outliers

# In[19]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')


# In[20]:


sns.distplot(data_1['Price'])


# In[21]:


sns.distplot(data_1['Mileage'])


# In[22]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[23]:


sns.distplot(data_2['Mileage'])


# In[24]:


data_3 = data_2[data_2['EngineV']<6.5]


# In[25]:


sns.distplot(data_3['EngineV'])


# In[26]:


sns.distplot(data_no_mv['Year'])


# In[27]:


q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]


# In[28]:


sns.distplot(data_4['Year'])


# In[29]:


data_cleaned = data_4.reset_index(drop=True)


# In[30]:


data_cleaned.describe(include='all')


# # Checking OLS Assumptions

# In[31]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()


# In[32]:


sns.distplot(data_cleaned['Price'])


# ### Relaxing Assumptions

# In[33]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price']=log_price
data_cleaned


# In[34]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')

plt.show()


# In[35]:


data_cleaned = data_cleaned.drop(['Price'],axis=1)


# ## Multicollinearity

# In[36]:


data_cleaned.columns.values


# In[37]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns


# In[38]:


vif


# In[39]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)


# # Create Dummy Variables

# In[40]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)


# In[41]:


data_with_dummies.head()


# ## Rearrenge

# In[42]:


data_with_dummies.columns.values


# In[43]:


cols = ['log_price','Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[44]:


data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# # Linear Regression Model

# ## Declare Inputs and Targets

# In[45]:


targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)


# In[46]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)


# In[47]:


inputs_scaled = scaler.transform(inputs)


# ## Train, Test, Split

# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


x_train, x_test, y_train, y_test= train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)


# ## Create the Regression

# In[50]:


reg = LinearRegression()
reg.fit(x_train, y_train)


# In[51]:


y_hat = reg.predict(x_train)


# In[53]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[54]:


sns.distplot(y_train - y_hat)
plt.title('Residuals PDF', size=18)


# In[55]:


reg.score(x_train, y_train)


# ## Finding weights and bias

# In[57]:


reg.intercept_


# In[56]:


reg.coef_


# In[58]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# In[60]:


data_cleaned['Brand'].unique()


# In[61]:


data_cleaned['Body'].unique()


# In[62]:


data_cleaned['Engine Type'].unique()


# # Testing

# In[63]:


y_hat_test = reg.predict(x_test)


# In[65]:


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[71]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns = ['Predictions'])
df_pf.head()


# In[74]:


y_test = y_test.reset_index(drop=True)
y_test.head()


# In[75]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns = ['Predictions'])
df_pf.head()


# In[80]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[83]:


df_pf['Prediction'] = np.exp(y_test)
df_pf


# In[84]:


df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']


# In[85]:


df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[86]:


df_pf.describe()


# In[88]:


pd.options.display.max_rows =999
pd.set_option('display.float_format', lambda x: '%.2f' %x)
df_pf.sort_values(by=['Difference%'])


# In[ ]:




