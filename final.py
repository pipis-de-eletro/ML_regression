#!/usr/bin/python3
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

fnx = "files/Xtrain_Regression1.npy"
fny = "files/Ytrain_Regression1.npy"
fnteste = "files/Xtest_Regression1.npy"
fnytest = "files/lasso/y_hat.npy"

def main():
    x = np.load(fnx)
    y = np.load(fny)
    
    '''
    ===================================================================================================================
    A big part of our procedure is missing, since they are not necessary to arrive at the model
    
    Brief resume of our procedure(a more detailed resume will be done in report):
    Separate the training set in a virtual training and testing set, train the model, test and evaluate it:
        -training/testing split: 80/20 samples
        -this is done so we can evaluate the model against overfitting and compare it against other model
        -comparisons were done with the r^2 and mse metrics
        -the alpha values for the cross validation were first run by cross-validation in logarithmic space
    ===================================================================================================================
    '''
    
    if x.shape[0] != y.shape[0]:
        print("Training set has different samples amounts for x and y.")
        exit(0)

    final_model = linear_model.LassoCV(alphas=np.linspace(0.001, 0.0035, 10)) 
    final_model.fit(x, y.ravel())

    print("MSE: %2f" % mse(y, final_model.predict(x)))
    print("Score: %2f" % final_model.score(x, y))
    
    x_test = np.load(fnteste)
    y_hat = final_model.predict(x_test) 
    
    np.save(fnytest, y_hat)

main()
