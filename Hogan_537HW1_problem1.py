from matplotlib import pyplot as plt
import scipy as sy
import numpy as np
import sympy as sp
import pandas as pd

np.set_printoptions(legacy='1.25')



#L = int(input("Input width of box: ")) # variable representing the length of the box
L=1

def b(x,j): # function that defines the Rayleigh-Ritz basis vectors
    b = x*(1-x)*x**j
    return b

def b_prime(k,coord):
    b_prime = sp.diff(b(x,k),x)
    bp = sp.lambdify(x, b_prime, "numpy")
    bp2 = bp(coord)
    return bp2
    
def psi(x, m, c):
    psi = 0
    for i in range(m):
        psi = psi + c[i]*b(x,i)
    return psi
        
# Solution to Problem 1
#### Task 1



#### Task 2

N = 200 # number of points in the G-L quadrature

def gaussLeg(c, d, n, i, j):
    [z, w] = sy.special.p_roots(n+1)
    G = 0.5*(d-c)*sum(w*b(0.5*(d-c)*z+0.5*(d+c), i)*b(0.5*(d-c)*z+0.5*(d+c), j))
    return G

def gaussLegDiff(c, d, n, i, j):
    [z, w] = sy.special.p_roots(n+1)
    G = 0.5*(d-c)*sum(w*b_prime(i, 0.5*(d-c)*z+0.5*(d+c))*b_prime(j, 0.5*(d-c)*z+0.5*(d+c)))
    return G

x=sp.Symbol('x') # allows for x to be treated as a variable of integration

M = int(input("How many Rayleigh-Ritz basis vectors?: "))

S = np.zeros((M, M)) # matrix of zeros as a blank slate for the S matrix
q = np.zeros((M,M))
for j in range(M): # loop that calculates S_ij
    for i in range(M):
        S[i][j] = gaussLeg(0, L, N, i, j)
        
print("S = ", S)

H = np.zeros((M, M))
p = np.zeros((M, M))
for j in range(M): # loop that calculates H_ij
    for i in range(M):
        bi_prime = sp.diff(b(x, i))
        bj_prime = sp.diff(b(x, j))
        H[i][j] = gaussLegDiff(0, L, N, i, j)
        
print("H = ", H)


#### Task 3
##solving the generalized eigenvalue problem Hc=E*Sc
eigvals, eigvecs = sy.linalg.eigh(H, S)
print("EIGVALS = ", eigvals)
print("EIGVECS = ", eigvecs)
lowestE = 1E99
for i in range(M):
    if eigvals[i]<lowestE:
        lowestE = eigvals[i]
print(lowestE)
print(str(lowestE), "- ", str(0.5*np.pi**2), "= ", lowestE - 0.5*np.pi**2)


#### Task 4

#trial wavefunction
fig, ax = plt.subplots(2,2)
xSpace = np.linspace(0, 1, 2000)
ax[0][0].set_ylim(0, 1.5)
ax[0][1].set_ylim(0, 1.5)
ax[1][0].set_ylim(0, 1.5E-5)
ax[0][0].plot(xSpace, np.sqrt(2)*np.sin(np.pi*xSpace), color = 'blue', label = 'Actual Eigenvector')
ax[0][1].plot(xSpace, psi(xSpace, M, eigvecs[:,0]), color = 'red', label = 'Approximated Eigenvector')
ax[1][0].plot(xSpace, abs(psi(xSpace, M, eigvecs[:,0]) - np.sqrt(2)*np.sin(np.pi*xSpace)), color = 'green', label = 'Absolute difference')
ax[1][1].axis('off')
fig.legend(loc = 'lower right')
ymax = max(abs(psi(xSpace, M, eigvecs[:,0]) - np.sqrt(2)*np.sin(np.pi*xSpace)))
xmax = xSpace[np.argmax(abs(psi(xSpace, M, eigvecs[:,0]) - np.sqrt(2)*np.sin(np.pi*xSpace)))]
ax[1][0].plot(xmax, ymax, 'go')
ax[1][0].annotate('(' + str(round(xmax,6)) + ', ' + str(round(ymax,6)) + ')', xy=(xmax, ymax),
         xycoords='data',
         xytext=(0.1+xmax, ymax), fontsize = 7,
         textcoords='data')

plt.show()


#### Task 5

