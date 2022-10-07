#!/usr/bin/python3
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

fnx = "files/Xtrain_Regression1.npy"
fny = "files/Ytrain_Regression1.npy"
fnbeta = "files/lasso/beta_aprox.npy"
fnteste = "files/Xtest_Regression1.npy"
fnytest = "files/lasso/y_hat.npy"

def main():
    x = np.load(fnx)
    y = np.load(fny)
    
    '''
    ========================================================================================================================
    The first two regressions were to aproximate alpha(with more precision) values, commented so they won't be used anymore
    Should try to automate it later
    ======================================================================================================================== 
    '''

    #reg = linear_model.LassoCV(alphas=np.logspace(-13, 13, 6))
    #reg = linear_model.LassoCV(alphas=np.linspace(0.0005, 0.005, 20))
    reg = linear_model.LassoCV(alphas=np.linspace(0.002, 0.004, 4))
    
    if x.shape[0] != y.shape[0]:
        print("Training set has different samples amounts for x and y.")
        exit(0)
    '''
    =========================================================================
    We separate the training set in a virtual training and testing set, 
    this is done so we can evaluate the model against overfitting and 
    compare it against other models
    I know this might not be necessary since we have cross-validation
    but since we can retrain it later w/ all the data, we decided to do it
    =========================================================================
    ''' 
    x_train = x[:-20]
    x_test = x[-20:]

    y_train = y[:-20]
    y_test = y[-20:]

    reg.fit(x_train, y_train.ravel())
    
    #np.save(fnbeta, reg.coef_) # non-necessary
    
    print(reg.alpha_)
     
    y_hat = reg.predict(x_test)
    print("Mean Squared Error = %2f" % mse(y_test, y_hat))
    print("Coefficient of determination = %2f" % r2(y_test, y_hat))
    
    '''
    ===========================================================================================================
    Retraining the model with all the data 
    ===========================================================================================================
    '''

    #final_model = linear_model.Lasso(alpha=reg.alpha_)
    final_model = linear_model.LassoCV(alphas=np.linspace(0.001, reg.alpha_, 10))
    final_model.fit(x, y.ravel())

    print("MSE: %2f" % mse(y, final_model.predict(x)))
    print("Score: %2f" % final_model.score(x, y))
    
    x_test = np.load(fnteste)
    y_hat = final_model.predict(x_test) 
    
    np.save(fnytest, y_hat)

main()
