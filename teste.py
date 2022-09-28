#!/usr/bin/python3
import numpy as np

fnx = "Xtrain_Regression1.npy"
fny = "Ytrain_Regression1.npy"

def main():
    x = np.load(fnx)
    y = np.load(fny) 
    
    if x.shape[0] != y.shape[0]:
        print("Training set has different samples amounts for x and y.")
        exit(0)

    n = x.shape[0]
    p = x.shape[1]
    print("n =", n, "; p =", p)
    
    xt = np.transpose(x)
    aux = np.linalg.inv(np.matmul(xt, x))
    print(aux.shape)
    aux2 = np.matmul(x, y)
    print(aux2.shape)
    beta = np.matmul(aux, aux2)
    print(beta.shape)

main()
