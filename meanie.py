#!/usr/bin/python3
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

fnx = "files/Xtrain_Regression1.npy"
fny = "files/Ytrain_Regression1.npy"
fnbeta = "results_mean/beta_aprox.npy"
fnteste = "files/Xtest_Regression1.npy"
fnytest = "results_mean/y_hat.npy"

def main():
    x = np.load(fnx)
    y = np.load(fny)
    
    aux = np.mean(x, axis=0)
    aux2 = np.broadcast_to(aux, (100, 10))
    x = x-aux2 

    aux = np.mean(y)
    y = y - aux

    if x.shape[0] != y.shape[0]:
        print("Training set has different samples amounts for x and y.")
        exit(0)

    x_train = x[:-20]
    x_test = x[-20:]

    y_train = y[:-20]
    y_test = y[-20:]
        
    xt = np.transpose(x_train)
    aux = np.linalg.inv(np.matmul(xt, x_train))
    
    aux2 = np.matmul(xt, y_train)
    
    beta = np.matmul(aux, aux2)
    
    np.save(fnbeta, beta)
    
    y_hat = np.matmul(x_test, beta)

    print("Mean Squared Error = %2f" % mse(y_test, y_hat))
    print("Coefficient of determination = %2f" % r2(y_test, y_hat)) 

    x_test = np.load(fnteste)

    y_hat = np.matmul(x_test, beta)
    np.save(fnytest, y_hat)


while True:
    a = input("b-calculate betas; t-generate testing data, e-exit: ")
    if a == "b":
        main()
    elif a == "e":
        exit(0)
    else:
        print("Please follow the instructions!")
