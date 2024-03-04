# main1b.py
# Linear regression (polynomial regression) on body density data.
# Cross-validation
#
# PMR Mar 2024

#%% Import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.model_selection as modelSelection

#%% Define polyExpand function
def polyExpand(X, k):
    '''Performs polynomial expansion of X as in Y = [X, X^2,...,X^k]
    X: pandas data frame
    k: polynomial order'''
    Y = X
    for n in range(2,k+1):
        XX = np.power(X,n)
        for c in XX.columns:
            XX.rename(columns={c:c+str(n)},inplace=True)
        Y = pd.concat([Y, XX], axis=1)
    return Y

#%% Import data
data = pd.read_csv('bodyMeasurementsSingleCV.txt')
I = data.columns == 'Body_Density'
y = data.iloc[:,I]
X = data.iloc[:,~I]

n = len(X)

#%% Run analysis

# Create random partition of data
K = 10
cGen = modelSelection.KFold(n_splits=K, shuffle=True, random_state=42)
c = list(cGen.split(X))

# Define polynomial orders
M = np.arange(1,8)

# Initialize arrays
errTrain = np.zeros((K,len(M)))
errTest = np.zeros((K,len(M)))

# Iterate over partitions
for idx1 in range(K):
    # Get training- and test sets
    idx_train = c[idx1][0]
    idx_test = c[idx1][1]
    #
    Xtrain = X.iloc[idx_train,:]
    ytrain = y.iloc[idx_train]
    Xtest = X.iloc[idx_test,:]
    ytest = y.iloc[idx_test]

    # Iterate over polynomial orders
    for idx2 in range(len(M)):
        
        # Create polynomial expansion of predictor variables
        XtrainPol = polyExpand(Xtrain,M[idx2])
        XtestPol = polyExpand(Xtest,M[idx2])
        
        # Scale predictor variables
        scl = XtrainPol.std()
        XtrainPol = XtrainPol/scl
        XtestPol = XtestPol/scl

        # Train linear regression model
        fit = LinearRegression().fit(XtrainPol, ytrain)
        
        # Use model to predict
        yhatTrain = fit.predict(XtrainPol)
        yhatTest = fit.predict(XtestPol)
        
        # Compute training and test error
        errTrain[idx1,idx2] = np.power(ytrain - yhatTrain, 2).mean().iloc[0]
        errTest[idx1,idx2] = np.power(ytest - yhatTest, 2).mean().iloc[0]

#%% Plot errors
fig, ax = plt.subplots()
ax.plot(M, errTrain.mean(axis=0), '-o', label = 'training')
ax.plot(M, errTest.mean(axis=0), '-*', label = 'test')
legend = ax.legend()
ax.set_xlabel('polynomial order')
ax.set_ylabel('mse')
ax.set_ylim([None, None]) # You might change y limits to zoom in
plt.show()