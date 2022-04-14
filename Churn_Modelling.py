#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn


# In[ ]:


tf.__version__


# In[ ]:


dataset=pd.read_csv('Churn_Modelling.csv')


# In[ ]:


dataset.head()


# In[ ]:


x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values


# In[ ]:


print(x)


# In[ ]:


print(y)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])


# In[ ]:


print(x)


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')


# In[ ]:


print(x)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[ ]:


ann=tf.keras.models.Sequential()


# In[ ]:


ann.add(tf.keras.layers.Dense(units=6,activation='relu'))


# In[ ]:


ann.add(tf.keras.layers.Dense(units=6,activation='relu'))


# In[ ]:


ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# In[ ]:


ann.compile(optimizer='adam',loss='bainary_crossentropy',metrics=['accuracy'])


# In[ ]:


ann.fit(x_train,y_train,batch_size=32,epochs=100)


# In[ ]:


print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))>0.5)


# In[ ]:


y_pred=ann.predict(x_test)
y_pred=(y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

