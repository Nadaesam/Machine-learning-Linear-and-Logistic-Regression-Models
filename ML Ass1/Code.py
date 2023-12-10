#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[2]:


data=pd.read_csv("D:/loan_old.csv")
data2=pd.read_csv("D:/loan_new.csv")


# In[3]:


print("Num of Missing Values each feature:\n")
print(data.isnull().sum())
print("\n")
NewData = data.dropna()
print("Data after deleting missing values:\n")
print(NewData)


# In[4]:


print("\nData Types:\n")
print(NewData.dtypes)


# In[5]:


num_features = NewData.select_dtypes(include=[np.number])
num_features_scale = num_features.std()
scaled_features_description = NewData[num_features.columns].describe()
print("Numerical Features Scale:\n")
print(num_features_scale)
print("\n")
print("Numerical Features description:\n")
print(scaled_features_description)
sns.pairplot(NewData[num_features.columns])
plt.show()


# In[6]:


X= NewData.drop(['Max_Loan_Amount', 'Loan_Status','Loan_ID'], axis=1)
y_amount = NewData['Max_Loan_Amount']
y_status = NewData['Loan_Status']
#Linear
print("\n Features:\n")
print(X.head(),"\n")
print("\nTarget Amount:\n")
print(y_amount.head(),"\n")
#logestic
print("\n Target Status:\n")
print(y_status.head())


# In[7]:


X_train,X_test, y_amount_train, y_amount_test, y_status_train, y_status_test = train_test_split(X, y_amount, y_status, test_size=0.1, random_state=42,shuffle=True)
print("X Train:" )
print(X_train,"\n")
print("X Test:")
print(X_test,"\n")
print("y_amount_Train:")
print(y_amount_train,"\n")
print("y_amount_Test:")
print(y_amount_test,"\n")
print("y_status_Train:")
print(y_status_train,"\n")
print("y_status_Test:")
print(y_status_test,"\n")


# In[11]:


label_encoder = LabelEncoder()
for column in ['Gender','Married','Education','Dependents','Property_Area','Credit_History' ]:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    X_test[column] = label_encoder.transform(X_test[column])

scaler = StandardScaler()
X_train[['Income','Coapplicant_Income','Loan_Tenor']]=scaler.fit_transform(X_train[['Income','Coapplicant_Income','Loan_Tenor']])
X_test[['Income','Coapplicant_Income','Loan_Tenor']]=scaler.fit_transform(X_test[['Income','Coapplicant_Income','Loan_Tenor']])

print("Encoded Standerlized Training Features (X_train):\n", X_train.head())
print("\nEncoded Standerlized Testing Features (X_test):\n", X_test.head())


# In[12]:


encoder = LabelEncoder()
y_status_train = encoder.fit_transform(y_status_train)
y_status_test = encoder.fit_transform(y_status_test)
print("Encoded numrical Target (Y_Status_Train):\n", y_status_train,"\n")
print("Encoded Standerlized Testing Features (Y_Status_test):\n", y_status_test)


# In[13]:


linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_amount_train)
y_amount_train_pred = linear_reg_model.predict(X_train)
# Evaluate the model using R2 score on the training set
r2_train = r2_score(y_amount_train, y_amount_train_pred)
print(f"R2 Score on Training Set: {r2_train}")


# In[18]:


# fit logistic regression 
import numpy as np
# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize theta array and alpha value and setting max iterations
theta = np.zeros(X_train.shape[1])
alpha = 0.01
max_iterations = 1000
# Perform gradient descent
m = len(y_status_train)
for iteration in range(max_iterations):
    z = np.dot(X_train, theta)
    h = sigmoid(z)

    gradient = np.dot(X_train.T, (h - y_status_train)) / m
    theta -= alpha * gradient


# In[19]:


# Make predictions on the test set
y_status_pred = sigmoid(np.dot(X_test, theta))
y_pred_class = np.where(y_status_pred >= 0.5, 1, 0)
# Calculate accuracy for logistic model
accuracy = np.mean(y_pred_class == y_status_test)
print("Accuracy:", accuracy) 
print("Accuracy in percentage :", accuracy*100)


# In[20]:


#  accuracy function
def accuracy(X,Y):
 return (np.sum(X==Y)/Y.size)*100
# test accuracy function  on data 
accuracy(y_pred_class,y_status_test)


# In[21]:


print("Num of Missing Values each feature:\n")
print(data2.isnull().sum())
print("\n")
newdata2 = data2.dropna()
print("Data after deleting missing values:\n")
print(newdata2)


# In[22]:


print("\nData Types:\n")
print(newdata2.dtypes)
num_features = newdata2.select_dtypes(include=[np.number])
num_features_scale = num_features.std()
scaled_features_description = newdata2[num_features.columns].describe()
print("Numerical Features Scale:\n")
print(num_features_scale)
print("\n")
print("Numerical Features description:\n")
print(scaled_features_description)
sns.pairplot(newdata2[num_features.columns])
plt.show()


# In[23]:


#ii
X = newdata2.drop('Loan_ID', axis=1) 
label_encoder = LabelEncoder()
for column in ['Gender','Married','Education','Dependents','Property_Area','Credit_History' ]:
    X[column] = label_encoder.fit_transform(X[column])
  
#vi
scaler = StandardScaler()
X[['Income','Coapplicant_Income','Loan_Tenor']]=scaler.fit_transform(X[['Income','Coapplicant_Income','Loan_Tenor']])

#Display the preprocessed data
print("Preprocessed Data:")
print(X.head())


# In[24]:


maxAmountPredicted = linear_reg_model.predict(X)
print("Max Amount:")
print(maxAmountPredicted)
loanPredicted = sigmoid(np.dot(X, theta))
lpClass = np.where(loanPredicted >= 0.5, 1, 0)
print("Loan Status Predicted:")
print(lpClass)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




