# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:29:16 2026

@author: 12242
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import PIL as pil


es = -5.34 # eV
h = -1.80 # eV 
s = 0.18 # overlap

H = np.array([[es, h],[h, es]]) # Hamiltonian matrix
S = np.array([[1, s],[s, 1]]) # Overlap matrix



eigvals, eigvecs = sp.linalg.eigh(H, S) # generalized eigenvalues and eigenvectors of Hc = E(Sc)

print('The eigenvalues are: ')
print(eigvals)
print('and the eigenvectors are: ')
print(eigvecs)
print('as column vectors.')
print()

print('To show that confirm that each eigenvector is normalized so its inner product on S is 1, calculate the inner product for each eigenvector and for the eigenvector matrix: ')
for i in range(len(eigvecs[:,0])): # confirming for normality fo individual eigenvectors
    eigT = np.transpose(eigvecs[:,i])
    print('vector' + str(i+1)+ ': ', eigT@S@eigvecs[:,i])
    
print('Matrix: ') # confirming normality of eigenvector matrix
print(np.transpose(eigvecs)@S@eigvecs)
print()

print('The one-electron energies are: ')

def Ep(t): # function calculating E+
    Ep = (es + h)/(1 + t)
    return Ep
def Em(t): # function calculating E-
    Em = (es - h)/(1 - t)
    return Em




print('E+ = (es + h)/(1+s) = ', round(Ep(s), 6))
print('E- = (es - h)/(1 - s)', round(Em(s),6))
print('Notice that these are the same as the eigenvalues.')
print()

S1 = eigvecs[:, 0] # naming the generalized eigenvectors for construction of symmetric and antisymmetric bonding states
S2 = eigvecs[:, 1]

bonding = (S1 + S2)/np.sqrt(2*(1 + s)) # bonding state
antibonding = (S1 - S2)/np.sqrt(2*(1 - s)) # antisymmetric bonding state

print('The symmetric bonding state is: ')
print('(S1 + S2)/sqrt(2(1 + s)) = ', bonding)
print('The symmetric bonding state is: ')
print('(S1 - S2)/sqrt(2(1 - s)) = ', antibonding)
print()


eigvals0, eigvecs0 = np.linalg.eigh(H) # solving the eigen-problem for S = I

S10 = eigvecs0[:, 0] # naming the eigenvectors for construction of symmetric and antisymmetric bonding states, S = I
S20 = eigvecs0[:, 1]

bonding0 = (S10 + S20)/np.sqrt(2) # bonding state, S = I
antibonding0 = (S10 - S20)/np.sqrt(2) # antibonding state, S = I


print('If S = 1, this is an ordinary eigenvalue problem, and the overlap is essentially neglected. The eigenvalues are')
print(eigvals0, ' , ')
print('the corresponding eigenvectors as columns are')
print(eigvecs0, ' , ')
print('and the bonding and antibonding states are')
print('S1 = ', bonding0)
print('S2 = ', antibonding0)
print('which are the canonical 2D basis vectors.')
print('The new one-electron energies are')
print('E+ = ', Ep(0))
print('E- = ', Em(0))
print('which are, once again, the same as the eigenvalues. They are noticeably more negative than without the orthogonal approximation though.')
print()


' plotting E+ and E- vs s(0.00, 0.40) with fixed es and h '
s0 = np.linspace(0.00, 0.40, 5000)
plt.plot(s0, Ep(s0), label = 'Bonding', color = 'green')
plt.plot(s0, Em(s0), label = 'Antibonding', color = 'blue')
plt.title("One-electron energies vs s-overlap")
' marking where s = 0.18 '
plt.plot(s, Ep(s), color = 'green', marker = 'o', markersize = 5)
plt.plot(s, Em(s), color = 'blue', marker = 'o', markersize = 5)
plt.ylabel('Energy in eV')
plt.xlabel('Overlap')

' saving as a 600-DPI png PIL object '
plt.savefig('HW2_P1.png', format = 'png', dpi = 600)
pil_img = pil.Image.open('HW2_P1.png')

plt.show()
#plt.show()