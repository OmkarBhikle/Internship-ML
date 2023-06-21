#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[3]:


df=pd.read_csv("C:/Users/Omkar/Desktop/Titanic-Dataset.csv")
df=df.drop("Embarked",axis=1)
df=df.drop("Cabin",axis=1)
df=df.drop("PassengerId",axis=1)
# le=LabelEncoder()
# x['Cabin'].fillna((df['Cabin'].mean()), inplace=True)
df['Age'].fillna((df['Age'].mean()), inplace=True)
df['Fare'].fillna((df['Fare'].mean()), inplace=True)
# print(df.isnull().sum())


# In[4]:


le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])

a=df['Survived']

from collections import Counter
print(Counter(a))


# In[5]:


#Important
x=df.drop("Name",axis=1)
x=x.drop("Ticket",axis=1)
x=x.drop("Survived",axis=1)
y=df['Survived']


# In[6]:


from imblearn.over_sampling import SMOTE
sms = SMOTE(random_state=0)
x,y = sms.fit_resample(x,y)
print(Counter(y))


# In[7]:


print(df['Age'])
Q1=df['Age'].quantile(0.25)
Q3=df['Age'].quantile(0.75)
IQR=Q3-Q1
print("IQR:",IQR)
upper=Q3+1.5*IQR
lower=Q1-1.5*IQR
print(upper)
print(lower)
out1=df[df['Age']<lower].values
out2=df[df['Age']>upper].values
df['Age'].replace(out1,lower,inplace=True)
df['Age'].replace(out2,upper,inplace=True)
print(df['Fare'])
Q1=df['Fare'].quantile(0.25)
Q3=df['Fare'].quantile(0.75)
IQR=Q3-Q1
print("IQR:",IQR)
upper=Q3+1.5*IQR
lower=Q1-1.5*IQR
print(upper)
print(lower)
out1=df[df['Fare']<lower].values
out2=df[df['Fare']>upper].values
df['Fare'].replace(out1,lower,inplace=True)
df['Fare'].replace(out2,upper,inplace=True)


# In[8]:


from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split

logr=LogisticRegression()
rfc=RandomForestClassifier()
dtc=DecisionTreeClassifier()
gbc=GradientBoostingClassifier()
#pca=PCA(n_components=2)
#pca.fit(x)
#x=pca.transform(x)
#print(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0,test_size=0.39)

func=[logr,rfc,dtc,gbc]

for item in func:
    item.fit(xtrain,ytrain)
    ypred=item.predict(xtest)
    a=(accuracy_score(ytest,ypred))
    print(accuracy_score(ytest,ypred))
    print(item,a*100,"%")


# In[ ]:




