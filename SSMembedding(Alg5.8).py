#%%

import numpy as np
from numpy import linalg as la
from scipy import linalg as si
from sympy import *
import math
import random
import time


#%%


def checkRate(Q):
    for i in range(4):
        for j in range(4):
            if Q[i][j] < 0 and i != j:
                return False
    return True


def ComplexEmbeddability(LogM, vaps, P, verbose=False):
    # number of possible generators bounded by det(M)
    # identify which are the complex eigenvalues and get V (with some tiny complex part due to computational errors)

    Pinv = la.inv(P)

    if vaps[3].imag != 0:
        if vaps[2].imag != 0:
            aux = P @ np.diag([0, 0, np.pi * 2j, -np.pi * 2j]) @ Pinv
        elif vaps[1].imag != 0:
            aux = P @ np.diag([0, np.pi * 2j, 0, -np.pi * 2j]) @ Pinv
        else:
            aux = P @ np.diag([np.pi * 2j, 0, 0, -np.pi * 2j]) @ Pinv
    elif vaps[2].imag != 0:
        if vaps[1].imag != 0:
            aux = P @ np.diag([0, np.pi * 2j, -np.pi * 2j, 0]) @ Pinv
        else:
            aux = P @ np.diag([0, np.pi * 2j, 0, -np.pi * 2j, 0]) @ Pinv
    else:
        aux = P @ np.diag([np.pi * 2j, -np.pi * 2j, 0, 0]) @ Pinv

    # Get the real V without imaginary component noise, this is necessary if det(M) is small
    V = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            V[i][j] = aux[i][j].real

    # We check all candidates Log(M)+2pik V according to the boundaries on k
    # The inicialization should be at infty -infty,
    # V is guaranteed to have at least one positive and one negative entry outside the diagonal, and we expect the first time this occurs to have better "values" than the initialization
    U = oo
    L = -oo
    epsilon = 0.00000000000000000000001

    for i in range(4):
        for j in range(4):
            if i != j:
                if V[i][j] > epsilon:
                    if -LogM[i][j] / V[i][j] > L:
                        L = -LogM[i][j] / V[i][j]
                elif V[i][j] < -epsilon:
                    if -LogM[i][j] / V[i][j] < U:
                        U = -LogM[i][j] / V[i][j]
    ##                elif LogM[i][j] < -epsilon:
    # print("M is not embeddable.")

    L = math.ceil(L)
    U = math.floor(U)

    # List of all generators
    if L < U:
        if verbose:
            print("M is embeddable but its rates are not identifiable.")
            print("Some of ts  Markov generators are:\n")
        while L <= U:
            print(LogM + L * V, ", with k=", L, "\n")
            L = L + 1
        return True
    elif L == U:
        if L != 0 and verbose:
            print("M is embeddable and its rates are identifiable.")
            print(
                "Its only Markov generator is:\n",
                LogM + L * V,
                "with determination k=",
                L,
                "\n",
            )
        return True
    else:
        # print( "M is not embeddable.\n")
        return False


#%%

# main

prec = 10000000  # fineness of the grid
power = 4
size = 10**power  # number of samples
n = 0
K3 = False
print(f" Running experiment with 10^{power} samples")

# Counters for different cases
Embed = 0
DLC = 0
DD = 0
repeated = 0
Complex = 0
sing = 0
DLCE = 0
DDE = 0
ComplexE = 0
errorCount = 0
n = 0
PosDet = 0
DeltaPlus = 0

InitialTime = time.time()

A = list(range(prec + 3))
while n < size:
    n = n + 1

    # we generate two rows of Markov matrices by sampling on a grid. The distance between consecutive nodes of the grid is  1/prec

    x, y, z = tuple(sorted(random.sample(A, 3)))
    a = x / prec
    b = (y - x - 1) / prec
    c = (z - y - 1) / prec
    d = (prec + 2 - z) / prec

    if K3:
        e = b
        f = a
        g = d
        h = c

    else:
        x, y, z = tuple(sorted(random.sample(A, 3)))
        e = x / prec
        f = (y - x - 1) / prec
        g = (z - y - 1) / prec
        h = (prec + 2 - z) / prec

    M = [[a, b, c, d], [e, f, g, h], [h, g, f, e], [d, c, b, a]]

    DLCflag = False
    negFlag = False
    DDflag = False
    complexFlag = False
    repeatedFlag = False
    errorFlag = False

    if (a >= 0.5 and f >= 0.5):  # Check if the matrix is Diagonally dominant (and hence also DLC)
        DD = DD + 1
        DLC = DLC + 1
        DDflag = True
        DLCflag = True

    elif ( a >= e and a >= h and a >= d and f >= b and f >= g and f >= c ):  # Check if the matrix is "Diagonal Largest in Column"
        DLCflag = True
        DLC = DLC + 1

    # Compute eigenvalues (vaps) and a basis of eigenvectors (matrix P).
    vaps, P = la.eig(M)  

    # We check wheter there is any negative or non-real eigenvalue.
    det = 1
    for i in range(4):  
        det = det * vaps[i]
        if vaps[i].imag != 0:
            complexFlag = True
        elif vaps[i] < 0:
            negFlag = True
        for j in range(4):
            if i != j and abs(vaps[i] - vaps[j]) == 0:
                repeatedFlag = True
    det = det.real


    # Counting eigenvalues configs
    if repeatedFlag:
        # the sample is removed
        n = n - 1
        repeated = repeated + 1
        print(
            "\n\n\n-------------------REPEATED MATRIX ---------------\n\n\n",
            M,
            "\n\n\n",
        )
        if DLCflag:
            DLC = DLC - 1
        if DDflag:
            DD = DD - 1
    else:
        if det > 0:
            PosDet = PosDet + 1

        if complexFlag:
            Complex = Complex + 1
        elif not negFlag:
            DeltaPlus += 1
            


        # Check embedabbility
        if negFlag:
            # non-repeated negative eigenvalue  => not embeddable.
            pass
        elif det == 0:
            # singular matrices have no logarithm (not embeddable).
            sing = sing + 1
        elif det > 0:
            LogM = si.logm(M)

            """ There might be small errors on the principal logarithm which lead to non-real entries with really small imaginary components. If that is the case the sample is discarded"""
            for i in range(4):
                for j in range(4):
                    if LogM[i][j].imag != 0:
                        errorFlag = True
            if errorFlag:
                errorCount = errorCount + 1
                n = n - 1

            # Testing the principal logarithm
            if not complexFlag:
                if checkRate(LogM):
                    Embed = Embed + 1
                    if DLCflag:
                        DLCE = DLCE + 1
                    if DDflag:
                        DDE = DDE + 1
            else:
                pass
                # Test all logs for complex case (INCLUDING THE PRINCIPAL LOG!)
                # if ComplexEmbeddability(LogM, vaps, P):
                #     Embed = Embed + 1
                #     if DLCflag:
                #         DLCE = DLCE + 1
                #     if DDflag:
                #         DDE = DDE + 1
                #     ComplexE = ComplexE + 1


print("\nNUMBER OF SAMPLES:","\n Total:",size,"\n", "Embeddable:",Embed,"\n", "DLC",DLC,"\n", "DD", DD,"\n", "Positive Determinant:", PosDet, "\n", "Complex eigenvalues:", Complex, "\nDelta Plus", DeltaPlus)

print("\nNUMBER OF EMBEDDABLE SAMPLES:\n","Total:",Embed, "\n DLC", DLCE,"\n DD:",DDE,"\n Complex",ComplexE)

print("\nRELATIVE VOLUME OF EMBEDDABLE\n", "SS", Embed/size, "\n DLC", DLCE/DLC,"\n DD:",DDE/DD,"\n Positive Determinant:", Embed/PosDet)


print("\n \nSpecial Cases \n Repeated and Singular:", repeated , sing)
print("Total time:",time.time()-InitialTime)
print("Error", errorCount)


#%%

from numpy.random import default_rng

rng = default_rng()


power = 7
size2 = 10**power

count = 0
delta = 0
DLC=0
DD=0

n = 0

print(f" Running experiment with 10^{power} CS samples with off-diagonal entries between 0 and 1")
while n < size2:

    complexFlag = False
    repeatedFlag = False
    negFlag = False
    n += 1
    # we generate two rows of Markov matrices by sampling on a grid. The distance between consecutive nodes of the grid is  1/prec
    row1 = rng.uniform(size=3)
    row2 = rng.uniform(size=3)
    if row1.sum() <= 1 and row2.sum() <= 1:
        count += 1

        a = 1 - row1.sum()
        b = row1[0]
        c = row1[1]
        d = row1[2]

        e = row2[0]
        f = 1 - row2.sum()
        g = row2[1]
        h = row2[2]
        M = [[a, b, c, d], [e, f, g, h], [h, g, f, e], [d, c, b, a]]

        
        vaps, P = la.eig(M)
        for i in range(
            4
        ):  # We check wheter there is any negative or non-real eigenvalue.
            if vaps[i].imag != 0:
                complexFlag = True
            elif vaps[i] < 0:
                negFlag = True
            for j in range(4):
                if i != j and abs(vaps[i] - vaps[j]) == 0:
                    repeatedFlag = True

        if not repeatedFlag and not negFlag and not complexFlag:
            delta += 1
            if a>=e and a>=h and a>=d and f>=b and f>=g and f>=c: #Check if the matrix is "Diagonal Largest in Column"
                DLC+=1
            if a>=0.5 and f>=0.5:
                DD+=1

print("Markov:", count, ",\nPositive eigenvalues (Delta+):", delta, "\nDLC:",DLC, "\nDD:",DD)

print("DLC/Delta+:", DLC/delta)
print("DD/Delta+:", DD/delta)

#%%

DeltaPlus