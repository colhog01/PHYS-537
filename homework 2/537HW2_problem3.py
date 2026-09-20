# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:10:56 2026

@author: 12242
"""

''' Problem 3: Slater-Koster matrix for an arbitrary bond '''
import PIL as pil
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import sympy as sm
import pandas as pd

R12 = [3.2, 2.3, 2.6] # directed bond vector

d = 0
for i in range(len(R12)):
    d = d + R12[i]**2
d = np.sqrt(d)
print('The bond length is d = ', round(d, 6))

print('The direction cosines are just the components of the unit bond vector.')

' calculating the unit bond vector '
Rhat = np.empty(len(R12))
for i in range(len(R12)):
    r = (d**-1)*R12[i]
    Rhat[i] = r

' reporting the direction cosines '
print('The unit bond vector is ', Rhat)
l = Rhat[0]
m = Rhat[1]
n = Rhat[2]
print('Then the direction cosines are l = ' + str(l) + ', m = ' + str(m) + ', and n = ' + str(n))
check = l**2 + m**2 + n**2 # checking that the sum of the squares of the direction cosines is 1 (or at least 0.999999999999)
print('The sum of the squares of the direction cosines should be one since they are components of a unit vector. The sum of the squares of the reported values is', check)

const = 7.62/(d**2) # hbar^2 / (me*d^2) in eV*A^2



' Harrison coefficients '
Vsso = -1.32*const
Vspo = 1.42*const
Vppo = 2.22*const
Vppn = -0.63*const

print('potentials = ', Vsso, Vspo, Vppo, Vppn)


matrix_labels = ['s', 'px', 'py', 'pz'] # basis

H12 = np.zeros(((len(Rhat)+1), (len(Rhat)+1))) # initializing the intersite block

for i in range(len(Rhat)+1): # adding values into the intersite block
    for j in range(len(Rhat)+1):
        if i==0 and j==0:
            H12[i][j] = Vsso
        elif i>0 and j==0:
            H12[i][j] = Vspo*Rhat[i-1]
        elif i==0 and j>0:
            H12[i][j] = -1*Vspo*Rhat[j-1]
        else:
            H12[i][i] = Vppn + (Vppo - Vppn)*(Rhat[i-1]**2)


df1 = pd.DataFrame(H12, columns = matrix_labels, index = matrix_labels) # intersite block with row, column labels
print("H12(R12) = ")
print(df1)
#print(H12)
reverse_bond = np.array([]) # initializing the reverse bond vector
for i in range(len(Rhat)):
    reverse_bond = np.append(reverse_bond, -1*Rhat[i])
    
H12_reverse = np.zeros(((len(Rhat)+1), (len(Rhat)+1))) # reverse-bond intersite block

for i in range(len(reverse_bond)+1): # adding values to the reverse-bond intersite block
    for j in range(len(reverse_bond)+1):
        if i==0 and j==0:
            H12_reverse[i][j] = Vsso
        elif i>0 and j==0:
            H12_reverse[i][j] = Vspo*reverse_bond[i-1]
        elif i==0 and j>0:
            H12_reverse[i][j] = -1*Vspo*reverse_bond[j-1]
        else:
            H12_reverse[i][i] = Vppn + (Vppo - Vppn)*(reverse_bond[i-1]**2)

df2 = pd.DataFrame(H12_reverse, columns = matrix_labels, index = matrix_labels) # reverse-bond intersite block with row, column labels
df3 = pd.DataFrame(np.transpose(H12), columns = matrix_labels, index = matrix_labels) # (not reverse bond) intersite block transposed and with row, column labels

print()
print("H12(-R12) = ")
print(df2)
print()
print('H12(R12) transposed = ')
print(df3)

if H12_reverse.all() == np.transpose(H12).all(): # checking that H(-R) = H(R)_transpose
    print('H12(-R12) = H12(R12)_transpose')
print()
for i in range(len(H12)):
    for j in range(len(H12)):
        if H12[i][j] == -1*H12_reverse[i][j] and H12[i][j] != 0:
            print("Element (" + matrix_labels[i] + ", " + matrix_labels[j] + ") changes sign under bond reversal.")

plt.imshow(H12, cmap = 'hot', interpolation = 'nearest') # heat map of intersite block

' saving heat map as 600-DPI png PIL object '
plt.savefig('HW2_P3.png', format = 'png', dpi = 600)
pil_img = pil.Image.open('HW2_P3.png')

plt.show()


