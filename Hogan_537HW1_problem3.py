# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 00:55:28 2026

@author: 12242
"""


"""
Packages needed to run the code
"""
from matplotlib import pyplot as plt
import scipy as sy
import numpy as np
import sympy as sp
import pandas as pd

np.set_printoptions(legacy='1.25')

J2eV = 6.242*(10**18)

"""
 Solution to Problem 3

"""

''' Task 1 '''

# graphing the F(x) function:

x1 = np.linspace(-10, 10, 1000) 

def F(x): # piecewise defining F(x)
    conditions = np.array([x==0, x==1, (x<0) | ((x>0) & (x<1)) | (x>1)])
    functions = [2, 1, lambda x: 1 + ((1 - x**2)/(2*x))*np.log(abs((1+x)/(1-x)))]
    F = np.piecewise(x, conditions, functions)
    return F

plt.plot(x1, F(x1))
plt.title('Plot of F(x) vs x')
plt.plot(1,1)
plt.show()


''' Task 2 '''


#### defining some important functions from the beginning of the problem ####

def sigmax(x): # exchange operator
    sigma = -(kF/np.pi)*F(x)
    return sigma
def eFree(x): # free electron energy
    ef = 0.5*((kF*x)**2)
    return ef
def eHF(x): # Hartree-Fock energy
    e = eFree(x) + sigmax(x)
    return e



#### plotting the energy dispersions: #### 

fig, ax = plt.subplots(2, 1)

x1 = np.linspace(0, 1.5, 1000)

energyData = [] # will be used for containing the requested values in the third task
energyLabels = ['kF', 'eFree', 'eHF', 'avg Kinetic', 'exchange E per e-', 'EHF/N'] # labels for the data table, also for the third task

for i in range(2):
    energies = []
    rs = i + 1 # solid state parameter
    kF = (1/rs)*((9*np.pi/4)**(1/3)) # Hartree-Fock wave number
    
    ### plotting free electron energy and Hartree-Fock energy dispersions 0 <= x(=k/kF) <= 1.5 ###
    ax[i].plot(x1, eFree(x1)*J2eV, label = "Free-Electron, rs = " + str(i+1), color = 'red') # free electron dispersion
    ax[i].plot(x1, eHF(x1)*J2eV, label = "Hartree-Fock, rs = " + str(i+1), color = 'blue') # HF dispersion
    ax[i].axvline(x = 1, linestyle = '--', color = 'black', label = "k = kF") # where k = kF (x = 1)
    ax[i].axvspan(0, 1, color = 'black', alpha=0.2) # occupied interval
    ax[i].legend(loc = 'upper left')
    
    ### energy values for each rs ###
    efree = eFree(1)*J2eV # free electron energy
    ehf = eHF(1)*J2eV # Hartree-Fock energy
    averageK = (3/10)*(kF**2)*J2eV # average kinetic energy
    exchangeE = -3*kF/(4*np.pi)*J2eV # exchange energy per electron
    EpN = averageK + exchangeE # E_HF / N
    energies = [f'{kF: 0.6e}', f'{efree: 0.6e}', f'{ehf: 0.6e}', f'{averageK: 0.6e}', f'{exchangeE: 0.6e}', f'{EpN: 0.6e}']
    energyData.append(energies)

print(energyData)

fig.supxlabel('x = k/kF')
fig.supylabel('Energy in eV')
fig.suptitle('Free-Electron and Hartree-Fock Orbital Dispersions')
plt.show()





''' Task 3 '''


#### making a data table graphic ####

fig, ax = plt.subplots()

# turning off axes so it doesn't look like a graph
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(energyData, columns=energyLabels, index = ['rs = 1', 'rs = 2']) # data table

ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center') # table labels


plt.show()




''' Task 4 '''


#### estimating derivative of Hartree-Fock energy at x = 1 ####


x0 = 1 # x-coordinate of interest, where k = kF
delta = [10**(-1), 10**(-2), 10**(-3), 10**(-4), 10**(-5), 10**(-6)] # list of differences in x


### calculating symmetric differences to estimate derivative ###
diffs = [] # empty list to contain all the differences in the Hartree-Fock energy, given delta
for i in range(len(delta)):
    diffs_i = []
    for j in range(10):
        rs0 = 2
        kF = (1/rs)*((9*np.pi/4)**(1/3))
        upper = eHF(1+(10-j)*delta[i])
        lower = eHF(1-(10-j)*delta[i])
        diff_i = (upper - lower)/(2*delta[i])
        
        diffs_i.append(diff_i)
    diffs.append(diffs_i)

print(diffs)
#### Task 5

