#!/usr/bin/python3
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

fnx = "files/Xtrain_Regression1.npy"
fny = "files/Ytrain_Regression1.npy"
fnbeta = "files/ridge/beta_aprox.npy"
fnteste = "files/Xtest_Regression1.npy"
fnytest = "files/ridge/y_hat.npy"

def main():
    x = np.load(fnx)
    y = np.load(fny)
    
    #reg = linear_model.RidgeCV(np.linspace(0.09, 0.13, 1000))
    reg = linear_model.Ridge(alpha=0.12)
    if x.shape[0] != y.shape[0]:
        print("Training set has different samples amounts for x and y.")
        exit(0)
    
    x_train = x[:-20]
    x_test = x[-20:]

    y_train = y[:-20]
    y_test = y[-20:]

    reg.fit(x_train, y_train)
    
    np.save(fnbeta, reg.coef_)
    
    #print(reg.alpha_)
     
    y_hat = reg.predict(x_test)
    print("Mean Squared Error = %2f" % mse(y_test, y_hat))
    print("Coefficient of determination = %2f" % r2(y_test, y_hat))

    x_test = np.load(fnteste)
    y_hat = reg.predict(x_test)
    
    np.save(fnytest, y_hat)

main()
