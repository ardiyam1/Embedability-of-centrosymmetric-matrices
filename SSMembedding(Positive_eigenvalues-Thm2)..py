#%%

import numpy as np
from numpy import linalg as la
from scipy import linalg as si
from sympy import *
import random
import time




#%%

def check_embeddable(lambdaa,mu,alpha_prime,beta_prime,alpha,beeta,disc,eigen_1,eigen_2,eigen_3): # Testing the embeddability if the SS Markov matrix has pairwise different positive eigenvalues.
    x = log(eigen_1)
    y = log(eigen_2)
    z = log(eigen_3)

    delta = ((y+z)+(alpha-beeta)*(y-z)/sqrt(disc))/2
    gamma = ((y+z)-(alpha-beeta)*(y-z)/sqrt(disc))/2
    epsilon = alpha_prime*(y-z)/sqrt(disc)
    phi = beta_prime * (y-z)/sqrt(disc)
    alpha_1 = x*(1-lambdaa)/(2-lambdaa-mu)
    beta_1 = x*(1-mu)/(2-lambdaa-mu)

    # print("delta:", delta)
    # print("gamma:", gamma)
    # print("epsilon:", epsilon)
    # print("phi:", phi)

    # print("eigen:",x,y,z)
    # print(alpha_1)
    # print(beta_1)



    if (abs(phi)<= -alpha_1) and (abs(epsilon)<= -beta_1) and (gamma<=alpha_1) and delta<=beta_1:
        return True
    return False

def checkRate(Q):
    for i in range(4):
        for j in range(4):
            if Q[i][j]< 0 and i!=j:
                return False
    return True

    
#%%   
#main

prec=1000000 #fineness of the grid
power = 6
size = 10**power  # number of samples
n=0

# Counters for different cases
Embed=0
T_Embed = 0
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

print(f" Running experiment with 10^{power} samples")
A = list(range(prec+3))
while n < size:
    n=n+1

   
    DLCflag= False
    DDflag=False
    errorFlag=False
    invalid_eigenvalues = True

    while invalid_eigenvalues:

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



        lambdaa = a+d
        mu = f+g
        eigen_1 = lambdaa + mu -1

        if eigen_1> 0:
            alpha_prime = e-h
            beta_prime = b-c
            alpha = f-g
            beeta = a-d
            disc = (alpha-beeta)**2 + 4*alpha_prime*beta_prime
            if disc > 0:
                eigen_2 = ((alpha+beeta) + sqrt(disc))/2
                eigen_3 = ((alpha+beeta) - sqrt(disc))/2
                if eigen_2>0 and eigen_3>0:
                    invalid_eigenvalues = False

    if a>=0.5 and f>=0.5: #Check if the matrix is Diagonally dominant (and hence also DLC)
        DD=DD+1
        DLC=DLC+1
        DDflag = True
        DLCflag = True


    elif a>=e and a>=h and a>=d and f>=b and f>=g and f>=c: #Check if the matrix is "Diagonal Largest in Column"
        DLCflag=True
        DLC=DLC+1

    if check_embeddable(lambdaa,mu,alpha_prime,beta_prime,alpha,beeta,disc,eigen_1,eigen_2,eigen_3): # Testing the embeddability if the SS Markov matrix has pairwise different positive eigenvalues.
        Embed=Embed+1
        if DLCflag:
            DLCE=DLCE+1
        if DDflag:
            DDE=DDE+1     

#     # LogM = si.logm(M)            
#     # if checkRate(LogM): # Testing the embeddability if the SS Markov matrix has pairwise different positive eigenvalues.
#     #     T_Embed+=1   

# print("Embeddable samples with thesis algorithm:", T_Embed)

print("\nNUMBER OF SAMPLES:","\n Total:",size,"\n", "Embeddable:",Embed,"\n", "DLC",DLC,"\n", "DD", DD,"\n", "Positive Determinant:", PosDet, "\n", "Complex eigenvalues:", Complex)

print("\nNUMBER OF EMBEDDABLE SAMPLES:\n","Total:",Embed, "\n DLC", DLCE,"\n DD:",DDE,"\n Complex",ComplexE)

print("\nRELATIVE VOLUME OF EMBEDDABLE\n", "SS", Embed/size, "\n DLC", DLCE/DLC,"\n DD:",DDE/DD,"\n\n")


# print("\n \nSpecial Cases \n Repeated and Singular:", Repeated , sing)
print("Total time:",time.time()-InitialTime)
# print("Error", errorCount)

# %%
