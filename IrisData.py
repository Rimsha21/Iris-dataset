
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.datasets import load_iris


# In[4]:


iris= load_iris()


# In[6]:


df=pd.DataFrame(iris.data, columns= iris.feature_names)
df.head()


# In[7]:


iris.target


# In[8]:


iris['target_names']


# In[9]:


type(iris['data'])


# In[10]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train, x_test,y_train,y_test = train_test_split(iris['data'], iris['target'],random_state=0)


# In[15]:


x_train.shape


# In[19]:


y_train.shape


# In[20]:


from sklearn.neighbors import KNeighborsClassifier


# In[21]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[22]:


knn.fit(x_train,y_train)


# In[24]:


y_pred = knn.predict(x_test)
y_pred


# In[27]:


prediction =knn.predict([[5.2,4.7,3.7,0.6]])
prediction


# In[28]:


iris['target_names'][prediction]


# In[29]:


knn.score(x_test,y_test)


# In[31]:


from sklearn.tree import DecisionTreeClassifier


# In[32]:


dtc = DecisionTreeClassifier()


# In[36]:


dtc.fit(x_train,y_train)


# In[37]:


dtc.predict(x_test)


# In[38]:


dtc.score(x_test,y_test)

