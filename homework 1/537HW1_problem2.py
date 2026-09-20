# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 00:50:52 2026

@author: 12242
"""

from matplotlib import pyplot as plt
import scipy as sy
import numpy as np
import sympy as sp
import pandas as pd

np.set_printoptions(legacy='1.25')


# Solution to Problem 2

N = 54
L = 1.00

''' Task 1 '''


#### enumerating all integer triples nx, ny, nz ####

principals = [] # empty array to hold the combinations of nx, ny, nz 
qvals = [] # empty array to hold all the q's
for i in range (17): # finding all the principal number triples and q's
    for j in range(17):
        for k in range (17):
            nx = 8
            ny = 8
            nz = 8
            nx = nx - i
            ny = ny - j
            nz = nz - k
            q = nx**2 + ny**2 + nz**2
            principals.append([nx, ny, nz])
            qvals.append(q)
print(qvals) # debugging
print(len(qvals)) #debugging
uniqueQ = [] # empty array to hold all the unique q values
grouped = [] # empty array used to group the integer triples by q
g = [] # empty array representing the orbital degeneracy function
for i in range(len(qvals)): # grouping triples by q
    if qvals[i] not in uniqueQ:
        uniqueQ.append(qvals[i])
        grouped.append([principals[i]])
        g.append(1)
    else:
        grouped[uniqueQ.index(qvals[i])].append(principals[i])
        g[uniqueQ.index(qvals[i])] = g[uniqueQ.index(qvals[i])] + 1
print(grouped) # debugging
print(uniqueQ) # debugging
print('unique qs', len(uniqueQ)) # debugging
print("g = ", g) # debugging

spinCap = [] # spin-inclusive capacity
for i in range(len(g)):
    spinCap.append(2*g[i])
print("spin-incl capacity = ", spinCap)

hbar = 1240/(2*np.pi) # eVnm/c
E0 = (1/(2*0.511*10**(3)))*(2*np.pi*hbar)**2

first12OccShell = []
first12OrbitDeg = []
first12SpinCap = []
first12DoS = []
for i in range(12):
    first12OccShell.append(uniqueQ[len(uniqueQ)-(i+1)])
    first12OrbitDeg.append(g[len(g)-(i+1)])
    first12SpinCap.append(spinCap[len(spinCap)-(i+1)])
    E = E0*first12OccShell[i]
    g3D = (1/(2*np.pi**2))*np.sqrt(E)*((2*0.511*10**3)/(hbar**2))**(3/2)
    first12DoS.append(g3D)
print('q = ', first12OccShell, 'g = ', first12OrbitDeg, 's = ', first12SpinCap)
print('g3Ds = ', first12DoS)



plt.stem(first12SpinCap, first12OccShell)
plt.xlabel('Shell Capacity')
plt.ylabel('q number')
plt.show()

plt.hist(first12DoS, bins = 10)
plt.ylim(0, 3.5)
plt.ylabel('Frequency')
plt.xlabel('Shell Capacaties')
plt.show()
# sortedprincipals = sorted(principals)
# print(sortedprincipals)

#### Task 2







#### Task 3
n=N/L
hbar = 1
me = 1
kf = (3*(np.pi**2)*n)**(1/3)
Ef = ((hbar*kf)**2)/(2*me)
percentDiff = round(abs(Ef-E0)/E0, 6)*100
print('kF =', kf)
print('Ef = ', Ef)
print('Percent difference from finite-cell energy: ', percentDiff)




#### Task 4



#### Task 5
