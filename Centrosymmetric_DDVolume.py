import numpy as np
from numpy import linalg as la
from scipy import linalg as si
from sympy import *
import math
import cdd
import random
import time


def GetRandomRow(fineness):
    x,y,z,s,t= tuple(sorted(random.sample(list(range(fineness+5)), 5)))
    a=x
    b=(y-x-1)
    c=(z-y-1)
    d=(s-z-1)
    e=(t-s-1)
    f=(fineness+4-t)
    return a,b,c,d,e,f

def checkRate(Q,MatrixSize):
    for i in range(MatrixSize):
        for j in range(MatrixSize):
        #    if Q[i][j].imag != 0: 
         #       return False
            if Q[i][j]< 0 and i!=j:
                return False

    return True



    
def WriteMatrix(f,A,rows):
    for i in range(rows):
        f.write(" %s \n" %A[i])

    


##============= MAIN ===============
def main():
    epsilon = 0.000000000000001 # x is considered to be 0 if |x|<10^-15
    dec = 3
    size= 100000  #Sample size

    n=0 #Sample counter
    Embed=0
    neg=0
    Sing=0
    #DLC=0
    #DD=0
    Repeated=0

    file=open("DDOutput.txt", "a+")
    file.write("\n========  New Run  ========  \n \nSample of %d diagonally dominant Markov matrices " %size)
    file.write("(entries with %d decimal numbers). \n" %dec)

    prec=int(10**dec/2)
    InitialTime=time.time()
    
    while n < size:
        n=n+1
        M= [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]    
        for i in range(3):
            M[i][0],M[i][1],M[i][2],M[i][3],M[i][4],M[i][5]= GetRandomRow(prec)
            M[i][i]=M[i][i]+prec

        for i in range(3):
            for j in range(6):
                M[3+i][5-j]=M[2-i][j]

        for i in range(6):
            rowsum=0
            for j in range(6):
                M[i][j]=M[i][j]/(2*prec)
                rowsum=rowsum+M[i][j]      
    
            
        if np.linalg.matrix_rank(M)<6:
            Sing=Sing+1
            file.write(" \nSingular Matrix:\n")
            WriteMatrix(file,M,6)
        else:#M has a real logarithm
            LogM = si.logm(M)
            if checkRate(LogM,6):
                Embed=Embed+1
                file.write("\nEmbeddable matrix:\n")
                WriteMatrix(file,M,6)
                file.write("\nIts principal logarithm is a generator:\n")
                WriteMatrix(file,LogM,6)
              
                    


    file.write("\n \nRESULTS:")

    file.write("\n %d embeddable matrices," %Embed)#"\n", "DLC",DLC,"\n", "DD", DD,"\n")

    potential= size-Sing-neg
    file.write(" of a total of %d potentially embeddable matrices. \n \n Embeddability was not tested for matrices with no real logarithm:" %potential)
    file.write("\n   - %d singular matrices" %Sing)
    file.write("\n   - %d matrices with a non-positive real eigenvalue (with multiplicity 1)"%neg)
   
    
    #file.write("\n %d matrices with complex eignevalues, " %Complex)
    #file.write(" %d of which are embeddable." %ComplexE)

    file.write("\n \n %d matrices where discarded and sampled again due to repeated eigenvalues." %Repeated)
    Ttime=time.time()-InitialTime
    file.write("\n Total time: %f seconds." %Ttime)
    file.write("\n \n \n")
    file.close()




main()
