# main1a.py
# Linear regression (polynomial regression) on body density data.
# Training- and test set.
#
# PMR Mar 2024

#%% Import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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
dataTrain = pd.read_csv('bodyMeasurementsSingleTrain.txt')
dataTest = pd.read_csv('bodyMeasurementsSingleTest.txt')

## Extract predictors and response variables from tables into X and y
I = dataTrain.columns == 'Body_Density'; # Identify column with response variable (body density)
ytrain = dataTrain.iloc[:,I]
Xtrain = dataTrain.iloc[:,~I]

I = dataTest.columns == 'Body_Density'; # Identify column with response variable (body density)
ytest = dataTest.iloc[:,I]
Xtest = dataTest.iloc[:,~I]

#%% Run analysis

# Create an x axis with high resolution for plotting the trained model
dum = pd.concat([Xtrain, Xtest], ignore_index=True, axis=0)
XHighRes = pd.DataFrame(np.linspace(dum.min(), dum.max(), 1000), columns=Xtrain.columns)

# Define polynomial order
k = 1

# Create polynomial expansion of predictor variables
XtrainPol = polyExpand(Xtrain,k)
XtestPol = polyExpand(Xtest,k)

# Also create polynomial expansion of data used for plotting the trained model
XhighResPol = polyExpand(XHighRes,k);

# Scale predictor variables
scl = XtrainPol.std()
XtrainPol = XtrainPol/scl
XtestPol = XtestPol/scl
XhighResPol = XhighResPol/scl

# Train linear regression model
fit = LinearRegression().fit(XtrainPol, ytrain)

# Use model to predict
yhatTrain = fit.predict(XtrainPol)
yhatTest = fit.predict(XtestPol)

# Compute training and test error
errTrain = np.power(ytrain - yhatTrain, 2).mean()
errTest = np.power(ytest - yhatTest, 2).mean()

#%% Plot the data, and the model predictions
fig, ax = plt.subplots()
ax.plot(XtrainPol.iloc[:,0], ytrain, 'o', label = 'training data')
ax.plot(XtestPol.iloc[:,0], ytest, '.', label = 'test data')
ax.plot(XhighResPol.iloc[:,0], fit.predict(XhighResPol), '-', label = 'fit')
legend = ax.legend()
ax.set_xlabel(XtrainPol.columns[0])
ax.set_ylabel(ytrain.columns[0])
ax.set_ylim([None, None]) # You might change y limits to zoom in
plt.show()
