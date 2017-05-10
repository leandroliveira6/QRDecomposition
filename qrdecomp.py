# -*- coding: utf-8 -*-
"""
Created on Wed May 10 02:52:12 2017

@author: Leandro
"""

import numpy as np
from math import sqrt
from numpy.linalg import norm

def householder(colj, inicio):
    n = len(colj)
    
    if(inicio==0):
        x = colj
    else:
        x = colj[inicio:n]
    
    e1 = np.zeros(len(x))
    e1[0] = 1
    norma = norm(x)
    v = x + x[0]*norma*e1/abs(x[0])

    I = np.eye(len(x))
    v = np.array([v]) #matriz com uma linha
    P = I - 2 * np.dot(np.transpose(v), v) / np.dot(v,np.transpose(v))

    return P


def givens(colj, inicio):
    n = len(colj)
    
    if(inicio==0):
        x = colj
    else:
        x = colj[inicio:n]
        
    n = len(x)
    R = np.eye(n)
    for k in range(n-1):
        xi = x[0]
        yi = x[k+1]
        
        coss = xi/(sqrt(xi**2 + yi**2))
        seno = -yi/(sqrt(xi**2 + yi**2))
        
        Rt = np.eye(n)
        Rt[0][0]     = coss
        Rt[0][k+1]   = -seno
        Rt[k+1][0]   = seno
        Rt[k+1][k+1] = coss
        
        R = np.dot(Rt, R)
        x = np.dot(Rt, x)
        
    return R

def decompQR(A, tipo):
    n = len(A)
    Q = np.eye(n)
    
    for j in range(n):
        colj = A[:,j]
        I = np.eye(n)
        
        if(tipo == "householder"):
            I[j:n,j:n] = householder(colj, j)
        elif(tipo == "givens"):
            I[j:n,j:n] = givens(colj, j)
            
        Q = np.dot(Q, np.transpose(I)) #I = Qp
        A = np.dot(I, A)
    
    R = A
    return Q, R

A = np.array([[12., -51.,   4.],
              [ 6., 167., -68.],
              [-4.,  24., -41.]])

B = np.array([[2., 3., 4.],
              [5., 6., 7.],
              [8., 9., 10.]])

C = np.array([[5., 1., 1.],
              [1., 5., 1.],
              [1., 1., 5.]])

print("Decomposição QR com reflexão de Householder:\n")
Q, R = decompQR(A, "householder")
print("Q={}\nR={}\n".format(Q.round(4),R.round(1)))

Q, R = decompQR(B, "householder")
print("Q={}\nR={}\n".format(Q.round(4),R.round(1)))

Q, R = decompQR(C, "householder")
print("Q={}\nR={}\n\n".format(Q.round(4),R.round(1)))

print("Decomposição QR com rotação de Givens:\n")
Q, R = decompQR(A, "givens")
print("Q={}\nR={}\n".format(Q.round(4),R.round(1)))

Q, R = decompQR(B, "givens")
print("Q={}\nR={}\n".format(Q.round(4),R.round(1)))

Q, R = decompQR(C, "givens")
print("Q={}\nR={}\n\n".format(Q.round(4),R.round(1)))