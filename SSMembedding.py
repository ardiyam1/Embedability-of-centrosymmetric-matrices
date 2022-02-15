import numpy as np
from numpy import linalg as la
from scipy import linalg as si
from sympy import *
import math
import cdd
import random
import time




def checkRate(Q):
    for i in range(4):
        for j in range(4):
            if Q[i][j]< 0 and i!=j:
                return False
    return True



def ComplexEmbeddability(LogM,vaps, P):
    #number of possible generators bounded by det(M)
    #identify which are the complex eigenvalues and get V (with some tiny complex part due to computational errors)

    
    Pinv = la.inv(P)

    
    if vaps[3].imag != 0:
        if vaps[2].imag != 0:
            aux = P@ np.diag([0,0,np.pi*2j,-np.pi*2j])@Pinv
        elif vaps[1].imag != 0:
            aux = P@ np.diag([0,np.pi*2j,0,-np.pi*2j])@Pinv
        else:
            aux = P@ np.diag([np.pi*2j,0,0,-np.pi*2j])@Pinv
    elif vaps[2].imag != 0:
        if vaps[1].imag != 0:
            aux = P@ np.diag([0,np.pi*2j,-np.pi*2j,0])@Pinv
        else:
            aux = P@ np.diag([0,np.pi*2j,0,-np.pi*2j,0])@Pinv
    else:
        aux = P@ np.diag([np.pi*2j,-np.pi*2j,0,0])@Pinv

    #Get the real V without imaginary component noise, this is necessary if det(M) is small
    V= np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            V[i][j]= aux[i][j].real
            
   
    #We check all candidates Log(M)+2pik V according to the boundaries on k
    #The inicialization should be at infty -infty,
    #V is guaranteed to have at least one positive and one negative entry outside the diagonal, and we expect the first time this occurs to have better "values" than the initialization   
    U=oo
    L=-oo
    epsilon = 0.00000000001

    
    for i in range(4):
        for j in range(4):
            if i!= j:
                if V[i][j]> epsilon:
                    if (-LogM[i][j]/V[i][j] > L):
                        L=-LogM[i][j]/V[i][j]
                elif V[i][j]<-epsilon:
                    if (-LogM[i][j]/V[i][j] < U):
                        U=-LogM[i][j]/V[i][j]
##                elif LogM[i][j] < -epsilon:
                    #print("M is not embeddable.")

    L= math.ceil(L)
    U= math.floor(U)

    #List of all generators
    if L < U:
        print( "M is embeddable but its rates are not identifiable.\n \nIts  Markov generators are:\n")
        while L <= U:
            print(LogM+L*V,", with k=", L,"\n")
            L = L+1
        return True
    elif L == U:
        if L!=0:
##        print( "M is embeddable and its rates are identifiable.\n Its only Markov generator is:\n",LogM+L*V, "with determination k=",L, "\n")
            return True
    else: 
        #print( "M is not embeddable.\n")
        return False





   
#main

prec=10000000 #fineness of the grid
size=100000    #number of samples
n=0


# Counters for different cases
Embed=0
DLC=0
DD=0
Repeated=0
Complex=0
sing=0
DLCE=0
DDE=0
ComplexE=0
errorCount=0
n=0
PosDet=0

InitialTime=time.time()


A = list(range(prec+3))
while n < size:
    n=n+1

    #we generate two rows of Markov matrices by sampling on a grid. The distance between consecutive nodes of the grid is  1/prec
    x,y,z= tuple(sorted(random.sample(A, 3))) 
    a=x/prec
    b=(y-x-1)/prec
    c=(z-y-1)/prec
    d=(prec+2 -z)/prec
    x,y,z= tuple(sorted(random.sample(A, 3)))
    e=x/prec
    f=(y-x-1)/prec
    g=(z-y-1)/prec
    h=(prec+2 -z)/prec
    M=[[a,b,c,d],[e,f,g,h],[h,g,f,e],[d,c,b,a]]

    DLCflag= False
    negFlag=False
    DDflag=False
    complexFlag=False
    repeatedFlag=False
    errorFlag=False
    
    if a>=0.5 and f>=0.5: #Check if the matrix is Diagonally dominant (and hence also DLC)
        DD=DD+1
        DLC=DLC+1
        DDflag = True
        DLCflag = True


    elif a>=e and a>=h and a>=d and f>=b and f>=g and f>=c: #Check if the matrix is "Diagonal Largest in Column"
        DLCflag=True
        DLC=DLC+1


    vaps, P = la.eig(M) #Compute eigenvalues (vaps) and a basis of eigenvectors (matrix P).

    det=1
    
    for i in range(4): # We check wheter there is any negative or non-real eigenvalue.
        det=det*vaps[i]
        if vaps[i].imag != 0:
            complexFlag=True
        elif vaps[i] < 0:
            negFlag=True
        for j in range(4):
            if i!=j and abs(vaps[i]-vaps[j])==0:
                repeatedFlag=True

    det = det.real
    if det>0:
        PosDet=PosDet+1

    if repeatedFlag: #if there is a repeated eigenvalue we are not able to test all the Markov generators candidates with this algorithm, in this case the sample is removed
        n=n-1
        repeated=repeated+1
        if DLCFlag:
            DLC=DLC-1
        if DDFlag:
            DD=DD-1
            
    elif negFlag: #if there is a non-repeated negative eigenvalue the matrix is not embeddable.
        if complexFlag:
            Complex=Complex+1
                
            
    elif det == 0: #singular matrices have no logarithm (not embeddable).
        sing=sing+1
        
    elif det>0:
        LogM = si.logm(M)

        #There might be small errors on the principal logarithm which lead to non-real entries with really small imaginary components. 
        for i in range(4): 
                for j in range(4):
                    if LogM[i][j].imag!=0:
                        errorFlag = True
        if errorFlag: #If that is the case the sample is discarded
            errorCount=errorCount+1
            n=n-1
            
        elif complexFlag: # Testing the embeddability if the SS Markov matrix has non-real eigenvalues
            Complex=Complex+1
            if ComplexEmbeddability(LogM,vaps,P):
                Embed=Embed+1
                if DLCflag:
                    DLCE=DLCE+1
                if DDflag:
                    DDE=DDE+1
                ComplexE=ComplexE+1
                
        elif checkRate(LogM): # Testing the embeddability if the SS Markov matrix has pairwise different positive eigenvalues.
            Embed=Embed+1
            if DLCflag:
                DLCE=DLCE+1
            if DDflag:
                DDE=DDE+1

        
print("\nNUMBER OF SAMPLES:","\n Total:",size,"\n", "Embeddable:",Embed,"\n", "DLC",DLC,"\n", "DD", DD,"\n", "Positive Determinant:", PosDet, "\n", "Complex eigenvalues:", Complex)

print("\nNUMBER OF EMBEDDABLE SAMPLES:\n","Total:",Embed, "\n DLC", DLCE,"\n DD:",DDE,"\n Complex",ComplexE)

print("\nRELATIVE VOLUME OF EMBEDDABLE\n", "SS", Embed/size, "\n DLC", DLCE/DLC,"\n DD:",DDE/DD,"\n Positive Determinant:", Embed/PosDet,"\n Complex",ComplexE/Complex)

print("\n \nSpecial Cases \n Repeated and Singular:", Repeated , sing)
print("Total time:",time.time()-InitialTime)
print("Error", errorCount)
