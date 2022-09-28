#!/usr/bin/python3
import numpy as np

fnx = "files/Xtrain_Regression1.npy"
fny = "files/Ytrain_Regression1.npy"
fnbeta = "files/beta_aprox.npy"
fnteste = "files/Xtest_Regression1.npy"
fnytest = "files/y_hat.npy"

def main():
    x = np.load(fnx)
    y = np.load(fny)
     
    if x.shape[0] != y.shape[0]:
        print("Training set has different samples amounts for x and y.")
        exit(0)

    n = x.shape[0]
    p = x.shape[1]
        
    xt = np.transpose(x)
    aux = np.linalg.inv(np.matmul(xt, x))
    
    aux2 = np.matmul(xt, y)
    
    beta = np.matmul(aux, aux2)
    
    np.save(fnbeta, beta)

def teste():
    beta = np.load(fnbeta)
    x_test = np.load(fnteste)

    y_hat = np.matmul(x_test, beta)
    print(y_hat.shape)
    np.save(fnytest, y_hat)

while True:
    a = input("b-calculate betas; t-generate testing data, e-exit: ")
    if a == "b":
        main()
    elif a == "t":
        teste()
    elif a == "e":
        exit(0)
    else:
        print("Please follow the instructions!")
