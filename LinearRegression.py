
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[6]:


import pandas as pd
bos = pd.DataFrame(boston.data)
print(bos.head())


# In[7]:


bos['PRICE'] = boston.target

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[9]:


# code source:https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()


# In[10]:


delta_y = Y_test - Y_pred;

import seaborn as sns;
import numpy as np;
sns.set_style('whitegrid')
sns.kdeplot(np.array(delta_y), bw=0.5)
plt.show()


# In[11]:


sns.set_style('whitegrid')
sns.kdeplot(np.array(Y_pred), bw=0.5)
plt.show()


# In[40]:


def sgd_predict(data_row, cofficients):
    predicted_value = 0
    for i in range(data_row.shape[0]):
        predicted_value += cofficients[i] * data_row.iloc[i]
        # print(data_row.iloc[i])
    predicted_value += cofficients[data_row.shape[0] - 1]
    return predicted_value

def sgd_optimise(data_x, data_y, l_rate, n_epoch):
    coefficients = [0.0 for i in range(X_train.shape[1] + 1)]
    for ep in range(n_epoch):
        sum_error = 0
        for i in range(data_x.shape[0]):
            pred = sgd_predict(data_x.iloc[i], coefficients)
            error = data_y.iloc[i] - pred
            sum_error += error**2
            # print("predicted:" + str(pred) + " " + str(error))
            t = len(coefficients)- 1
            for j in range(t):
                coefficients[j] = coefficients[j] - \
                                    (error * l_rate * data_x.iloc[i].iloc[j]) 
                print(coefficients[j])
            coefficients[t] = coefficients[t] - (error * l_rate)
        print("epoch: %d, l_rate=%.5f, error=%.3f" 
              %(ep, l_rate, sum_error))
    return coefficients


# In[41]:


coefficients = sgd_optimise(X_train, Y_train, -0.0000001, 100)


# In[38]:


sgd_predict(X_train.iloc[5], coefficients)


# In[39]:


X_train.iloc[5]


# In[33]:


X_train.iloc[6].iloc[7]

