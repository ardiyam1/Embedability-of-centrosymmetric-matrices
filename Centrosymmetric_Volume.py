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
            if Q[i][j]< 0 and i!=j:
                return False
    return True




def OtherLogarithms(L,V,epsilon): #given L and V, check if there exists an integeer k s.t. L+kV is a rate matrix.
    #V has at least one positive and one negative off-diagonal entry by construction, so both up and low are bounded.
    up=oo
    low=-oo
    for i in range(6):
        for j in range(6):
            if i!= j:
                if V[i][j]> epsilon:
                    if (-L[i][j]/V[i][j] > low):
                        low=-L[i][j]/V[i][j]
                elif V[i][j]<-epsilon:
                    if (-L[i][j]/V[i][j] < up):
                        up=-L[i][j]/V[i][j]
                elif L[i][j] < -epsilon: #The set N is not empty
                    return False
    low= math.ceil(low)
    up= math.floor(up)
    if low<=up:
        return True
    return False



def ObtainBranchMatrix(P,Posi, Negi):
    Pinv = la.inv(P)
    if Posi == 0:
        if Negi == 1:
            aux = P@ np.diag([np.pi*2j,-np.pi*2j,0])@Pinv
        else:
            aux = P@ np.diag([np.pi*2j,0,-np.pi*2j])@Pinv
    elif Posi == 1:
        if Negi == 2:
            aux = P@ np.diag([0,np.pi*2j,-np.pi*2j])@Pinv
        else:
            aux = P@ np.diag([-np.pi*2j,np.pi*2j,0])@Pinv
    else:
        if Negi == 0:
            aux = P@ np.diag([-np.pi*2j,0,np.pi*2j])@Pinv
        else:
            aux = P@ np.diag([0,-np.pi*2j,np.pi*2j])@Pinv
    V= np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            V[i][j]= aux[i][j].real
    return V

    
def WriteMatrix(f,A,rows):
    for i in range(rows):
        f.write(" %s \n" %A[i])
    f.write("\n")

    
def checkDLC(M):
    for j in range(6):
        for i in range(6):
            if M[i][j]>M[j][j]:
                return False
    return True

##============= MAIN ===============
def main():
    epsilon = 0.000000000000001 # x is considered to be 0 if |x|<10^-15. This is used to compute L,U and N when using propositions 7.2 and 7.4
    dec = 4
    size= 1000000  #Sample size

    n=0 #Sample counter
    Embed=0
    neg=0
    Sing=0
    DLC=0
    DD=0
    Repeated=0
    DLCE=0
    DDE=0
    caseA=0
    caseB=0
    caseC=0
    caseD=0
    caseAE=0
    caseBE=0
    caseCE=0
    caseDE=0
    
    


    

    acc=10**dec #Rows are obtained by uniformly distributing acc units among the 6 entries.Each unit corresponds to the minimal decimal considered, i.e. 1/acc
    DDbound = acc/2
    InitialTime=time.time() #we want to know the running time so we can get an estimator of the running time for larger samples.
    

    #Open files to save the embeddable matrices 
    EAfile=open("EmbeddableMatrices(Case1).txt", "a+")
    EBfile=open("EmbeddableMatrices(Case2).txt", "a+")
    ECfile=open("EmbeddableMatrices(Case3).txt", "a+")
    EDfile=open("EmbeddableMatrices(Case4).txt", "a+")
    
    while n < size:


        #We generate a 6x6 centrosymmetric Markov matrices by picking a point (according to the uniform distr) in the grid of matrices with only %dec decimal numbers or less in each entry
        M= [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]    
        for j in range(3):
            M[j][0],M[j][1],M[j][2],M[j][3],M[j][4],M[j][5]= GetRandomRow(acc)
        for i in range(3):
            for j in range(6):
                M[3+i][5-j]=M[2-i][j]
  
          

        #Compute  eigenvalues and eigenvectors of each block of the Fourier transform (denoted here by U and L)
        UcomplexFlag = False
        LcomplexFlag = False

        # Upper Block
        U=[[M[0][0]+M[0][5], M[0][1]+M[0][4], M[0][2]+M[0][3]],[M[1][0]+M[1][5], M[1][1]+M[1][4], M[1][2]+M[1][3]],[M[2][0]+M[2][5], M[2][1]+M[2][4], M[2][2]+M[2][3]]]
        Uval, UP = la.eig(U)

        # Lower Block
        L=[[M[2][2]-M[2][3], M[2][1]-M[2][4], M[2][0]-M[2][5]],[M[1][2]-M[1][3], M[1][1]-M[1][4], M[1][0]-M[1][5]], [M[0][2]-M[0][3], M[0][1]-M[0][4], M[0][0]-M[0][5]]]
        Lval, LP = la.eig(L)

        
    
        #If there are repeated eigenvalues we discard this sample point
        RepeatedFlag=False
        vaps = [Uval[0],Uval[1],Uval[2],Lval[0],Lval[1],Lval[2]]
        det=1
        for i in range(6): 
            for j in range(6):
                if vaps[i]==vaps[j] and i!=j:
                    RepeatedFlag=True
        if RepeatedFlag:
            Repeated=Repeated+1


        else:
            n=n+1

            # Check wheter M is a DLC or a Diagonally dominant matrix
            DLCflag= False
            DDflag=False
            if  M[0][0]>=DDbound and M[1][1]>=DDbound and M[2][2]>=DDbound and M[3][3]>=DDbound and M[4][4]>=DDbound and M[5][5]>=DDbound:
                DD=DD+1
                DLC=DLC+1
                DDflag = True
                DLCflag = True
            elif checkDLC(M): 
                DLCflag=True
                DLC=DLC+1

            #If M is a singular matrix it is not embeddable    
            if np.linalg.matrix_rank(M)<6: 
                Sing=Sing+1

            #Check if M has non-positive or non-real eigenvalues.     
            else:   
                NegFlag=False
                #Upper block. Uposi and Unegi denote the position of the non-real eigenvalues (if any) with positive and negative imaginary part 
                for i in range(3): 
                    if Uval[i].imag > 0:
                        UcomplexFlag=True
                        UPosi= i
                    elif Uval[i].imag < 0:
                        UNegi = i
                    elif Uval[i] <= epsilon:
                        NegFlag = True
                    Uval[i]=Uval[i]
                #Same for the LowerBlock
                for i in range(3): 
                    if Lval[i].imag > 0:
                        LcomplexFlag=True
                        LPosi= i
                    elif Lval[i].imag < 0:
                        LNegi = i
                    elif Lval[i] <= epsilon:
                        NegFlag = True
                    Lval[i]=Lval[i]

                # If M has a negative eigenvalue (with multiplicity 1) it has no real logarithm and is not embeddable.
                if NegFlag: 
                    neg=neg+1

                #Otherwise, M has a real logarithm with rows summing to 0
                else:
                    
                    #Cases
                    if UcomplexFlag and LcomplexFlag:
                        caseD = caseD+1
                    elif UcomplexFlag:
                        caseB = caseB+1
                    elif LcomplexFlag:
                        caseC = caseC+1
                    else:
                        caseA=caseA+1

           

                    # Check if Log(M) is a rate matrix
                    for i in range(6):
                        for j in range(6):
                            M[i][j]=M[i][j]/acc

                    LogM = si.logm(M)
                    LogM=LogM.real
                    if checkRate(LogM,6):
                        Embed=Embed+1
                        

                        if UcomplexFlag and LcomplexFlag:
                            caseDE = caseDE+1
                            WriteMatrix(EDfile,M,6)
                        elif UcomplexFlag:
                            caseBE = caseBE+1
                            WriteMatrix(EBfile,M,6)
                        elif LcomplexFlag:
                            caseCE = caseCE+1
                            WriteMatrix(ECfile,M,6)
                        else:
                            caseAE=caseAE+1
                            WriteMatrix(EAfile,M,6)
                        if DLCflag:
                            DLCE=DLCE+1
                        if DDflag:
                            DDE=DDE+1
                       
                    # If M lies in cases 2,3,4 it may have a Markov generator different than Log(M)
                   
                    elif UcomplexFlag: # Case 3 & 4
                        Aux = ObtainBranchMatrix(UP,UPosi, UNegi)
                        Aux=Aux/2
                        UV=[[Aux[0][0],Aux[0][1],Aux[0][2],Aux[0][2],Aux[0][1],Aux[0][0]],[Aux[1][0],Aux[1][1],Aux[1][2],Aux[1][2],Aux[1][1],Aux[1][0]],[Aux[2][0],Aux[2][1],Aux[2][2],Aux[2][2],Aux[2][1],Aux[2][0]],[Aux[2][0],Aux[2][1],Aux[2][2],Aux[2][2],Aux[2][1],Aux[2][0]],[Aux[1][0],Aux[1][1],Aux[1][2],Aux[1][2],Aux[1][1],Aux[1][0]],[Aux[0][0],Aux[0][1],Aux[0][2],Aux[0][2],Aux[0][1],Aux[0][0]]]
                        LOne= LogM-UV

                        #Embeddability test for matrices in case 3
                        if not LcomplexFlag: 
                            if checkRate(LOne,6):
                                Embed=Embed+1
                                WriteMatrix(ECfile,RealM,6)
                                caseCE=caseCE+1
                                if DLCflag:
                                    DLCE=DLCE+1
                                if DDflag:
                                    DDE=DDE+1
                                
                        #Embeddability test for matrices in case 4      
                        else:   
                            Aux = ObtainBranchMatrix(LP,LPosi, LNegi)
                            Aux=Aux/2
                            LV=[[ Aux[2] [2], Aux[2] [1], Aux[2] [0],- Aux[2] [0],- Aux[2] [1],- Aux[2] [2]],[ Aux[1] [2], Aux[1] [1], Aux[1] [0],- Aux[1] [0],- Aux[1] [1],- Aux[1] [2]],[ Aux[0] [2], Aux[0] [1], Aux[0] [0],- Aux[0] [0],- Aux[0] [1],- Aux[0] [2]],[- Aux[0] [2],- Aux[0] [1],- Aux[0] [0], Aux[0] [0], Aux[0] [1], Aux[0] [2]],[- Aux[1] [2],- Aux[1] [1],- Aux[1] [0], Aux[1] [0], Aux[1] [1], Aux[1] [2]],[- Aux[2] [2],- Aux[2] [1],- Aux[2] [0], Aux[2] [0], Aux[2] [1], Aux[2] [2]]]
                            #caseD= caseD+1
                            if OtherLogarithms(LogM,LV,epsilon) or OtherLogarithms(LOne,LV,epsilon):
                                Embed=Embed+1
                                WriteMatrix(EDfile,RealM,6)
                                caseDE=caseDE+1
                                if DLCflag:
                                    DLCE=DLCE+1
                                if DDflag:
                                    DDE=DDE+1
                                

                    #Embeddability test for matrices in case 2
                    elif LcomplexFlag: 
                        Aux = ObtainBranchMatrix(LP,LPosi, LNegi)
                        Aux=Aux/2
                        LV=[[ Aux[2] [2], Aux[2] [1], Aux[2] [0],- Aux[2] [0],- Aux[2] [1],- Aux[2] [2]],[ Aux[1] [2], Aux[1] [1], Aux[1] [0],- Aux[1] [0],- Aux[1] [1],- Aux[1] [2]],[ Aux[0] [2], Aux[0] [1], Aux[0] [0],- Aux[0] [0],- Aux[0] [1],- Aux[0] [2]],[- Aux[0] [2],- Aux[0] [1],- Aux[0] [0], Aux[0] [0], Aux[0] [1], Aux[0] [2]],[- Aux[1] [2],- Aux[1] [1],- Aux[1] [0], Aux[1] [0], Aux[1] [1], Aux[1] [2]],[- Aux[2] [2],- Aux[2] [1],- Aux[2] [0], Aux[2] [0], Aux[2] [1], Aux[2] [2]]]
                        if OtherLogarithms(LogM,LV,epsilon):
                            Embed=Embed+1
                            WriteMatrix(EBfile,RealM,6)
                            caseBE= caseBE+1
                            if DLCflag:
                                DLCE=DLCE+1
                            if DDflag:
                                DDE=DDE+1
                            

                    
                    


    #print Output and close files containint embeddable matrices               
    potential= size-Sing-neg
    Ttime=time.time()-InitialTime

    
    file=open("Output.txt", "a+")
    file.write("\n========  New Run  ========  \n \n Sample of %d Markov matrices " %size)
    file.write("(entries with %d decimal numbers): \n" %dec)
    file.write("   - %d Markov matrices with real logarithms with rows summing to zero.\n" %potential)
    file.write("   - %d DLC Markov matrices. \n" %DLC)
    file.write("   - %d diagonally dominant Markov matrices. \n" %DD)
    
    file.write("\n %d embeddable matrices:" %Embed)
    file.write("\n   - %d embeddable DLC matrices." %DLCE)
    file.write("\n   - %d embeddable diagonally dominant matrices.\n" %DDE)

    
    file.write("\n Eigenvalues:\n   - %d embeddable matrices," %caseAE)
    file.write(" of %d Markov matrices in case 1." %caseA)
    file.write("\n   - %d embeddable matrices," %caseBE)
    file.write(" of %d Markov matrices in case 2." %caseB)
    file.write("\n   - %d embeddable matrices," %caseCE)
    file.write(" of %d Markov matrices in case 3." %caseC)
    file.write("\n   - %d embeddable matrices," %caseDE)
    file.write(" of %d Markov matrices in case 4." %caseD)
    file.write("\n   - %d matrices with a simple non-positive real eigenvalue (not embeddable)."%neg)
    file.write("\n   - %d singular matrices (not embeddable)." %Sing)
    

    file.write("\n \n %d matrices where discarded and sampled again due to repeated eigenvalues." %Repeated)
    file.write("\n Total time: %f seconds." %Ttime)
    file.write("\n \n \n")
    file.close()
    
    EAfile.close()
    EAfile.close()
    EAfile.close()
    EAfile.close()
    
main()
