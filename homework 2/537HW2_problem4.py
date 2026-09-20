# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:34:18 2026

@author: 12242
"""

import PIL as pil
import numpy as np
from matplotlib import pyplot as plt

eOs = -34.02 # eV
eOp = -16.77 # eV


d = [1.208] # list to contain the several O-O separations
for i in range(10):
    d0 = 1.208 + 0.005*(i+1)
    d1 = 1.208 - 0.005*(i+1)
    d.append(round(d0, 6))
    d.append(round(d1, 6))
d.sort()
print('distances = ', d)


def Harrison(dist): # calculting the Harrison coefficients for each bond distance
    const = 7.62 # eV*A^2
    Vsso = -1.32*const/dist
    Vspo = 1.42*const/dist
    Vppo = 2.22*const/dist
    Vppn = -0.63*const/dist
    return Vsso, Vspo, Vppo, Vppn

def Hamiltonian(dist): # create a Hamiltonian matrix for each bond distance. Also returns eigen-solutions, the diagonalized Hamiltonian, and a hermiticity check
    Vsso, Vspo, Vppo, Vppn = Harrison(dist)
    
    matrix = np.array([[eOs, 0, 0, 0, Vsso, 0, 0, Vspo], [0, eOp, 0, 0, 0, Vppn, 0, 0], 
         [0, 0, eOp, 0, 0, 0, Vppn, 0], [0, 0, 0, eOp, -1*Vspo, 0, 0, Vppo], 
         [Vsso, 0, 0, -1*Vspo, eOs, 0, 0, 0], [0, Vppn, 0, 0, 0, eOp, 0, 0], 
         [0, 0, Vppn, 0, 0, 0, eOp, 0], [Vspo, 0, 0, Vppo, 0, 0, 0, eOp]]) # the Hamiltonian
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix) # eigenvalues and eigenvectors for the Hamiltonian
    
    diagonalized_matrix = np.linalg.inv(eigenvectors)@matrix@eigenvectors # the diagonalized Hamiltonian
    
    if matrix.all() == np.transpose(matrix).all(): # hermiticity check (is H = to H_transpose?)
        hermiticity = True
    else:
        hermiticity = False
    
    return matrix, diagonalized_matrix, eigenvalues, eigenvectors, hermiticity

def bondsAndOrbitals(dist): # fills orbitals, writes the orbital configuration, classifies eigenstates, calculates bond order and Eband
    Vsso, Vspo, Vppo, Vppn = Harrison(dist)
    eigenvals, eigenvects = Hamiltonian(dist)[2], Hamiltonian(dist)[3]
    bond_energies = np.array([0.5*(eOs+Vsso+eOp-Vppo) + np.sqrt(0.25*(eOs+Vsso-eOp+Vppo)**2 + Vspo**2),
                      0.5*(eOs+Vsso+eOp-Vppo) - np.sqrt(0.25*(eOs+Vsso-eOp+Vppo)**2 + Vspo**2), 
                      0.5*(eOs-Vsso+eOp+Vppo) + np.sqrt(0.25*(eOs-Vsso-eOp-Vppo)**2 + Vspo**2), 
                      0.5*(eOs-Vsso+eOp+Vppo) - np.sqrt(0.25*(eOs-Vsso-eOp-Vppo)**2 + Vspo**2), 
                      eOp + Vppn, eOp - Vppn]) # calculation of the bond energies, without degeneracy of pi-bond
    
    "classifying the eigenstates"
    bond_labels = np.array(['sigma', 'sigma', 'sigma*', 'sigma*', 'pi', 'pi*']) 
    classification = np.array([])
    for j in range(len(eigenvals)):
        for k in range(len(bond_energies)):
            if np.isclose(eigenvals[j], bond_energies[k], rtol = 1e-6) == True:
                classification = np.append(classification, str(eigenvects[:, j]) + 'is a ' + bond_labels[k] + ' bond')
                #print(eigenvects[:, j], 'is a ' + bond_labels[k] + ' bond')
    
    sorted_labels = bond_labels[bond_energies.argsort()] # sorts the labels based on bond energy
    
    filled_orbitals = [] # list to represent the orbitals. 1 space for sigma, two spaces for pi
    for j in range(len(sorted_labels)):
        if 'sigma' in sorted_labels[j]:
            filled_orbitals.append([0])
        if 'pi' in sorted_labels[j]:
            filled_orbitals.append([0, 0])
            
    valence = 12 # total valence electrons in O-O
    for k in range(len(filled_orbitals)): # places the electrons into orbitals by the Pauli principle and Hund's rule
        while all(x < 2 for x in filled_orbitals[k]) and valence > 0:
            for l in range(len(filled_orbitals[k])):
                filled_orbitals[k][l] = filled_orbitals[k][l] + 1
                valence = valence - 1
    
    "writing out the molecular orbital configuration based on the orbital filling"
    atomic_orbitals = np.array(['2s', '2s', '2pz', '2px', '2py', '2px', '2py', '2pz']) # subscripts
    sorted_labels = np.insert(sorted_labels, slice(3, 5), ['pi', 'pi*']) # adding back degenerate pi states
    orbital_labels = sorted_labels + '_' + atomic_orbitals # the possible molecular orbitals
    exponents = np.array([]) # number of electrons in each
    for j in range(len(filled_orbitals)):
        for k in range(len(filled_orbitals[j])):
            exponents = np.append(exponents, filled_orbitals[j][k])
    mol_orbit_config = '' 
    configList = []
    num_bonding_electrons = 0
    num_antibonding_electrons = 0
    for j in range(len(exponents)):
        if exponents[j] > 0:
            mo = '(' + orbital_labels[j] + ')' + '^' + str(int(exponents[j]))
            mol_orbit_config = mol_orbit_config + mo
            configList.append(mo)
            if '*' in mo:
                num_antibonding_electrons = num_antibonding_electrons + exponents[j]
            else:
                num_bonding_electrons = num_bonding_electrons + exponents[j]
    
    bond_order = 0.5*(num_bonding_electrons - num_antibonding_electrons) # calculates the bond order
    
    Eband1 = 0 # initializing band energy
    eigenvals.sort()
    for j in range(len(exponents)): # calculating band energy
        
        Eband1 = Eband1 + eigenvals[j]*exponents[j]
    
    return mol_orbit_config, filled_orbitals, bond_order, classification, Eband1


band_energies = np.array([])

for i in range(len(d)): 
    Vsso, Vspo, Vppo, Vppn = Harrison(d[i])
    H, D, eigvals, eigvecs, herm = Hamiltonian(d[i])
    print("The raw Hamiltonian is ")
    print(H)
    print()
    print('The diagonalized Hamiltonian is ')
    print(D)
    print('eigenvectors = ', eigvecs)
    print('eigenvalues = ', eigvals)
    
    print()
    
    print('Hermiticity check (H =? H_dagger): Since this matrix is real', 
          ' its conjugate transpose is just its transpose,'
          ' so check to see that the Hamiltonian is equal to its transpose.')

    print('H = H_transpose?: ', herm)
    
    print()
    
    print("Classifying the eigenstates as sigma, sigma*, pi, or pi*:")
    
    print()
    
    MO_config, orbitalFilling, bondOrder, classes, band_energy = bondsAndOrbitals(d[i])
    print()
    for j in classes:
        print(j)
    print()
    print("The molecule's orbitals fill like: ")
    print(orbitalFilling)
    
    print()
    
    print("This corresponds to this molecule's molecular orbital configuration:")
    print(MO_config)
    print("where the single-element sub-arrays in the array represent sigmas, and the two-element sub-arrays represent pis.")
    
    print()
    
    print("The bond order is calculated as half the difference between the number of bonding electrons and the number of antibonding electrons. This molecule's bond order is ", bondOrder, ".")
    if bondOrder == 2:
        print("The bond order is 2, which corresponds to a double bond.")
    elif bondOrder == 1:
        print("The bond order is 1, which corresponds to a single bond.")
    else:
        print("The bond order is " + str(bondOrder))
    
    print()
    
    num_unpaired = 0 # variable for counting number of unpaired electrons
    for k in MO_config: # calculating number of unpaired electrons
        if k == '1':
            num_unpaired = num_unpaired + 1
    if num_unpaired > 0:
        print("This O2 molecule also has " + str(num_unpaired) + " unpaired electrons, so it is paramagnetic.")
    elif num_unpaired == 0:
        print("This O2 molecule also has no unpaired electrons, so it is diamagnetic.")
    
    print()
    
    band_energies = np.append(band_energies, band_energy)
    print("The band energy for this bond distance is " + str(round(band_energy, 6)))

Esep = 2*(2*eOs + 4*eOp) # separation energy, eV
print("Esep = ", Esep)

print("To find what C is in Vrep, we can set the derivative of Etot at d0 equal to zero. This forces the derivative of Vrep to be equal in size to the derivative of Eband at that bond distance. "
      "So dEband/dd = 4C/(d^5). To estimate dEband/dd, we can take finite differences about d0.")


" Calculating C "
sym_diffs = np.array([]) # list to hold the symmetric differences from 1.208 angstrom
for i in range(len(d)):
    if i > 0 and i < len(d)-1:
        diff_i = (band_energies[i+1] - band_energies[i-1])/(2*(d[i+1] - d[i-1]))
        sym_diffs = np.append(sym_diffs, diff_i)
print(sym_diffs)
print("Since this is a symmetric difference, the central term in the array will be the closest to the actual value. To solve for C, multiply this number by d0^5.")
C = sym_diffs[int(0.5*(len(sym_diffs)-1))]*(d0**5) # the coefficient in Vrep
print('dEband/dd', sym_diffs[int(0.5*(len(sym_diffs)-1))])
print('C = ', round(C, 6))
print()
print("Then Vrep = C/d^4.")
print("Then Etot = Eband + Vrep, and delta-E = Esep - Etot. Both of these functions are also functions of bond distance, as are Vrep and Eband. Esep is fixed since eOs and eOp are fixed.")
print("Now we can plot these with respect to distance.")

print()

" plotting Eband - Esep, Etot, and delta-E, as well as finding Ecoh(deq) "
# these arrays to hold values of each type of energy for different bond distances
plotEband = np.array([])
plotEtot = np.array([])
plotDelE = np.array([])
Ecoh = np.array([])
bond_distance = np.linspace(1, 10, 5000)

for i in bond_distance: # filling out the energy arrays
     plotEband = np.append(plotEband, bondsAndOrbitals(i)[4])
     plotEtot = np.append(plotEtot, bondsAndOrbitals(i)[4] + (C/(i**4)))
     plotDelE = np.append(plotDelE, bondsAndOrbitals(i)[4] + (C/(i**4)) - Esep)

# plotting the energies for Task 5
plt.plot(bond_distance, plotEband - Esep, label = 'Eband - Esep')
plt.plot(bond_distance, C/(bond_distance**4), label = 'Vrep')
plt.plot(bond_distance, plotDelE, label = 'delta-E')
plt.title('Various Energies vs Bond Distance')
plt.xlabel('Bond Distance (angstrom)')
plt.ylabel('Energy (eV)')

' saving plot as 600-DPI png PIL object '
plt.savefig('HW2_P4.png', format = 'png', dpi = 600) # saving plot as 600-DPI png
pil_img = pil.Image.open('HW2_P4.png') # opening as a pillow object

plt.show()

for i in plotEtot:
    Ecoh = np.append(Ecoh, Esep - i)

plt.plot(bond_distance, Ecoh, color = 'blue')


deq = bond_distance[np.argmax(Ecoh)] # equilibrium distance based on max cohesive energy
print('The equilibrium bond distance is ' + str(round(deq,6)))
print('and the model cohesive energy for this distance is ' + str(round(max(Ecoh).real, 6)))

plt.plot(deq, Ecoh[np.argmax(Ecoh)], 'bo')
plt.title('Cohesive Energy vs Bond Distance')
plt.ylabel('Cohesive Energy (eV)')
plt.xlabel('Bond Distance (angstrom)')
plt.show()
























