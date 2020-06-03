"""
Created on Fri May 15 15:54:25 2020
@author: DESHMUKH
Support Vector Machine 
"""
# pip install keras
# pip install tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import normalize
pd.set_option('display.max_columns',None)

# =================================================================================================================
# Business Problem - Prepare support vector machines model for classifying the area under fire for foresfires data.
# =================================================================================================================

ff = pd.read_csv('fireforests.csv')
ff.head()
ff = ff.iloc[:,2:]
ff.info()
ff.isnull().sum()

# Summary
ff.describe()

# Histogram
ff.hist(grid = False)

# Boxplot
ff.boxplot(notch = True, patch_artist = True, grid = False);plt.xticks(fontsize=6, rotation = 90)

# Pairplot
sns.pairplot(ff.iloc[:,0:9], corner = True, diag_kind = "kde")

# Heat map and Correlation Coifficient 
sns.heatmap(ff.iloc[:,0:9].corr(), annot = True, annot_kws={"size": 6}, cmap = 'Oranges')

# WE CAN OBSERVE FROM ABOVE HEATMAP THERE IS NO CORRELATION IN BETWEEN OUTPUT AND INPUT FEATURES.
# SO WE CAN'T ABLE TO GET PROPER RESULT WITH HELP OF THIS INPUT FEATURE ONLY. 

###################################### - Data Preprocessing - #######################################

# Nomalization of data (as data contain binary value)
ff.iloc[:,0:8] = normalize(ff.iloc[:,0:8])

##################################### - Splitting data - ############################################

# Splitting in X and y
X = ff.drop(['area'],axis=1)
y = ff['area']

# Splitting in Train and Test 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3)

######################################### - Fitting Model- ##########################################

# Fitting SVM Model
model = SVR(kernel = 'rbf', gamma = 1, epsilon = 0.01)
model.fit(X_train,y_train)
      
# Accuracy of Model on Train and Test data
model.score(X_train,y_train)
model.score(X_test,y_test)

# Predication Train data and Test data
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# RMSE Train data and Test data
rmse_train = np.sqrt(np.mean((pred_train-y_train)**2))
rmse_test = np.sqrt(np.mean((pred_test-y_test)**2)) 

# Visualising Train data and Test data
plt.plot(pred_train,y_train,"bo")
plt.plot(pred_test,y_test,"ro")

                                # ---------------------------------------------------- #
