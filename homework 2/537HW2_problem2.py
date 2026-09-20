# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:21:34 2026

@author: 12242
"""
import PIL as pil
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import sympy as sm

''' Exact singlet of the Hubbard dimer '''

e = 0
t = 1
u = [0, 1, 2, 4, 8, 16]
Et = 2*e
U = []
for i in range(len(u)):
    U.append(u[i]*t)

def Hamiltonian(Uvalue): # function to create a Hamiltonian based on U
    matrix = np.array([[2*e, -2*t],[-2*t, 2*e + Uvalue]])
    return matrix

print('If we set epsilon and t to be 0 and 1, respectively, then the Hamiltonian will be')
for i in range(len(U)): # making a Hamiltonian for each U/t
    H = Hamiltonian(U[i])
    print('for U/t = ' + str(u[i]), ', ')
    print(H)

print('and the energies will be ') # reporting Et and J
print('Et = ', Et, ' (energy of the triplet)')
print('J = ', Et, ' - Egs (singlet-triplet splitting, Egs is the ground-state energy)')


print()
ground_states = np.array([]) # array to hold Egs for all assigned U/t
print('Diagonalizing all the Hamiltonians gives')
for i in range(len(U)):
    H = Hamiltonian(U[i]) # Hamiltonian
    eigvals, eigvecs = np.linalg.eig(H) # eigen-solutions
    D = np.linalg.inv(eigvecs)@H@eigvecs # diagonalized Hamiltonian
    print('for U/t = ' + str(u[i]), ', ')
    print(D)
    print(H)
    print()
    print('eigenvalues = ', eigvals)
    Egs = 2*e + 0.5*U[i] - np.sqrt(4*(t**2) + 0.25*(U[i]**2)) # calculating Egs
    print('Egs = ', round(Egs, 8))
    ground_states = np.append(ground_states, Egs) # adding Egs to collection of ground state energies
    print()
print()
print('Notice that the ground-state energy is exactly the lowest eigenvalue for each Hamiltonian, which is what the ground-state energy should be.')
print('Since t = 1, Egs/t = Egs in magnitude. And J = -Egs when epsilon = 0 => J/t = -Egs/t = -Egs')
print("The ionic probability |c_I|^2 = |sin(theta)|^2 is a trig function where the angle is dependent on U.")
print("theta = arctan((sqrt(U^2 + (4t)^2)/)/(4t))")
for i in range(len(U)):
    print("For U/t = " + str(u[i]) + ",")
    print("Egs/t = ", round(ground_states[i]/t, 6))
    print("J/t = ", round((e - ground_states[i])/t, 6))
    angle = np.arctan(np.sqrt(U[i]**2 + (4*t)**2)/(4*t)) # calculating the angle to calculate the ionic probability
    c_I = (np.sin(angle))**2 # calculating ionic probability
    print("The ionic probability for this U is ", round(c_I, 6))
    print()

print("The large-U superexchange estimate is given by 4t^2 / U.")
for i in range(len(U)):
    if U[i] > 0:
        J = e - ground_states[i] # calculating J
        estimate = 4*t**2 / U[i] # calculating the large-U superexchange estimate
        print("J for this U is ", J, "and the superexchange for this U is ", estimate)
        error = 100*abs(J - estimate)/J # comparison of J and superexchange
        print("The estimate is " + str(round(error, 6)) + "% from the exact J")

' creating a two-panel figure of J and the superexchange over U/t[0.25, 20] '
x = np.linspace(0.25, 20, 1000) # U/t from 0.25 to 20, inclusive

plotIonicProb = np.array([]) # array to contain all the ionic probability values to be plotted
for i in x: # calculating each ionic probability
    theta = np.arctan((np.sqrt((i*t)**2 + (4*t)**2)-t*i)/(4*t))
    plotIonicProb = np.append(plotIonicProb, np.sin(theta))

fig, ax = plt.subplots(2, 1)
ax[0].plot(x, e/t - (2*e/t + 0.5*x - np.sqrt(4 + 0.25*(x**2))))
ax[0].set_ylabel('J/t and 4t/U')
ax[0].set_xlabel('U/t')
ax[0].plot(x, 4/x)
ax[1].plot(x, plotIonicProb)
ax[1].set_ylabel('Ionic Probability')
ax[1].set_xlabel('U/t')

' saving plot as a 600-DPI png PIL object '
plt.savefig('HW2_P2.png', format = 'png', dpi = 600)
pil_img = pil.Image.open('HW2_P2.png')

plt.plot()

