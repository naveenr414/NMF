import numpy as np
from util import makeRandom
from copy import deepcopy

def e(beta):
    if(beta<=2):
        return 1/(3-beta)
    return 1/(beta-1)

def gamma(beta):
    if(beta<1):
        return 1/(2-beta)
    if(1<=beta<=2):
        return 1
    return 1/(beta-1)

def d(beta,x,y):
    if(beta==1):
        return x * np.log(x/y)/np.log(10) - x + y
    if(beta==0):
        return x/y - np.log(x/y)/np.log(10)-1
    return x**beta/(beta*(beta-1)) + y**beta/beta - (x*y**(beta-1))/(beta-1)

def D(beta,V,WH):
    return sum([d(beta,V.item(i),WH.item(i)) for i in range(V.size)])

def p(k,n,W,V,VH,beta):
    return sum([W[f,k]*V[f,n]*VH[f,n]**(beta-2) for f in range(W.shape[0])])

def q(k,n,W,H,V,VH,beta):
    return sum([W[f,k]*VH[f,n]**(beta-1) for f in range(W.shape[0])])

def ard(V,k,a,beta,tau,phi,max_iter=300,ltype="l2"):
    EPSILON=10**-6
    INFINITY = 10000000
    
    #Defining the dimensions of the matrix
    f = V.shape[0]
    n = V.shape[1]

    #Track changes in lambda through time (Lambda is matrix of size k)
    lambdaList = np.zeros((max_iter,k))

    #Add small values of epsilon to prevent divide by zero errors
    V=np.add(V,EPSILON)

    meanV = np.sum(V)/(f*n)

    #Why initialize like this?
    W = (makeRandom(f,k)+1)*(meanV**.5/k)
    H = (makeRandom(k,n)+1)*(meanV**.5/k)
    b = np.pi * (a-1)*meanV/(2*k)

    c = f+n+a+1
    if(ltype=="l2"):
        c = (f+n)/2 + a+1
        
    lamb = (np.sum(W**2,axis=0).T/2 + np.sum(H**2,axis=1)/2 + b)/c
    lambdaList[0] = lamb
    tol = INFINITY

    expfunc = gamma
    if(ltype=="l2"):
        expfunc = e

    iteration = 1
    while(tol>tau and iteration<max_iter):
        #Update H
        top = W.T.dot((W.dot(H)**(beta-2)) * V)
        bottom = W.T.dot(W.dot(H)**(beta-1))
        if(ltype=="l1"):
            bottom+=phi/np.tile(np.array([lamb]).T,(1,n))
        elif(ltype=="l2"):
            bottom+=phi*H/np.tile(np.array([lamb]).T,(1,n))
        
        allElements=H>0
        #Checking for NaN
        if(((top/bottom)**(expfunc(beta)))[0,0]!=((top/bottom)**(expfunc(beta)))[0,0]):
            break
        H[allElements] = H[allElements]*((top/bottom)**expfunc(beta))[allElements]

        #Update W
        top = (W.dot(H)**(beta-2)*V).dot(H.T)
        bottom = (W.dot(H)**(beta-1)).dot(H.T)
        if(ltype=="l1"):
            bottom+=phi/np.tile(np.array([lamb]),(f,1))
        elif(ltype=="l2"):
            bottom+=phi*W/np.tile(np.array([lamb]),(f,1))

        allElements=W>0
        #Checking for NaN
        if(((top/bottom)[0,0]**(expfunc(beta)))!=((top/bottom)**(expfunc(beta)))[0,0]):
            break
        W[allElements] = W[allElements] * ((top/bottom)**expfunc(beta))[allElements]

        exp = 1
        if(ltype=="l2"):
            exp=2
        lamb = (np.sum(W**exp,axis=0).T/2 + np.sum(H**exp,axis=1)/2 + b)/c
        lambdaList[iteration] = lamb
        previous = lambdaList[iteration-1]
        tol = max((lamb-previous)/previous)

        #To avoid division by Zero problems, I'm not sure if this is needed
        if(0 in W or 0 in H):
            iteration = max_iter
            
        iteration+=1

    #Avoid divide by zero errors elsewhere 
    W+=EPSILON
    H+=EPSILON
    return (W,H)

        
