#!/usr/bin/env python
# coding: utf-8

# # PROJECT CODE

# # MUHAMMAD UMAIR____01-1314161-046

# 

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:




dataset = pd.read_csv('D:/Bachelors Degree/BS CS-7A/Data mining/assignments/winequalityred.csv')  


# In[5]:


dataset.head()


# In[19]:


dataset.describe()


# In[20]:


x = dataset.drop('quality', axis=1)  
y = dataset['quality'] 


# In[21]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=0)  


# In[22]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:





# In[23]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
a=clf_entropy.fit(X_train, y_train)


# In[ ]:





# In[24]:


y_pred_en = clf_entropy.predict(X_test)
df=pd.DataFrame({'Actual quality':y_test,'ID 3':y_pred_en})  
df 


# In[25]:


plt.scatter(y_test,y_pred_en)


# In[26]:


from sklearn.metrics import accuracy_score
print("Accuracy with ID3 is ", accuracy_score(y_test,y_pred_en)*100)


# In[27]:


from sklearn import tree


# # ROOT node of following tree is alchohol

# In[30]:


r=tree.export_graphviz(a,label='root')
print('------------>complete tree is given below<--------------------------------------')


# In[31]:


print(r)

