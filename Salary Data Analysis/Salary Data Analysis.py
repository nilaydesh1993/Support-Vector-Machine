"""
Created on Wed May 13 12:18:18 2020
@author: DESHMUKH
Support Vector Machine
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.svm import SVC
pd.set_option('display.max_columns',None)

# ===============================================================================================
# Business Problem - Prepare a classification model using SVM for salary data.
# ===============================================================================================

salary_train = pd.read_csv("SalaryData_Train.csv")
salary_test = pd.read_csv("SalaryData_Test.csv")
salary_train.info()
salary_test.info()
salary_test.head()

# Value counts
salary_train.groupby('workclass').size()
salary_train.groupby('education').size()
salary_train.groupby('maritalstatus').size()
salary_train.groupby('occupation').size()
salary_train.groupby('relationship').size()
salary_train.groupby('race').size()
salary_train.groupby('sex').size()
salary_train.groupby('native').size()

############################ - Converting input variable into LabelEncoder - ############################

# All input categorical columns
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

# Counverting into Numerical Data by means of custome function and Lable Encoder
for i in string_columns:
    number = LabelEncoder()
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

######################################### - Data Preprocessing - ########################################

from sklearn.preprocessing import scale
# Nomalization of data (as data contain binary value)
salary_train.iloc[:,0:13] = scale(salary_train.iloc[:,0:13])
salary_test.iloc[:,0:13] = scale(salary_test.iloc[:,0:13])

##################################### - Spliting data in X and y - ######################################

X_train = salary_train.iloc[:,0:13]
y_train = salary_train.iloc[:,13]
X_test  = salary_test.iloc[:,0:13]
y_test  = salary_test.iloc[:,13]

########################################### - Fitting Model - ###########################################
# Kernel option - 'linear', 'poly', 'rbf', 'sigmoid'

# Fitting SVM Model
model = SVC(kernel = "linear")
model.fit(X_train,y_train)

# Accuracy
model.score(X_train,y_train) 
model.score(X_test,y_test) 

# Prediction on Train & Test Data
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Confusion matrix of Train and Test
## Train
confusion_matrix_train = pd.crosstab(y_train,pred_train,rownames=['Actual'],colnames= ['Train Predictions']) 
sns.heatmap(confusion_matrix_train, annot = True, cmap = 'Blues',fmt='g')

## Test
confusion_matrix_test = pd.crosstab(y_test,pred_test,rownames=['Actual'],colnames= ['Test Predictions']) 
sns.heatmap(confusion_matrix_test, annot = True, cmap = 'Greens',fmt='g')

# Classification Report of test
print(classification_report(y_test,pred_test))


                          # ---------------------------------------------------- #
