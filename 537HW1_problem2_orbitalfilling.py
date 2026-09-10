# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:36:45 2026

@author: 12242
"""
import numpy as np
from matplotlib import pyplot as plt

N = 54 # number of electrons 
fig, ax = plt.subplots()

n = [1, 2, 3, 4, 5] # possible principal numbers for 54 gs electrons
l = [] # list to contain orbital numbers
ml = [] # list to contain magnetic orbital numbers
ms = [1/2, -1/2] # possible spins, in case we need it
for i in range(len(n)): # listing out all the orbital and magnetic orbital numbers for a given n
    l.append(i)
    group = []
    for j in range(l[i]+1):
        group.append(-j)
        if j not in group:
            group.append(j)
    group.sort() # sorting the ml subsets
    ml.append(group)

orbitals = ['s', 'p', 'd', 'f', 'g'] # orbital labels

# debugging
print("n = ", n)
print("l = ", l)
print("ml = ", ml)

# some initial values and lists for creating the graph 
elevation_n = 0.5 # base elevation of the s-orbitals
lineStart1 = 0 # start of the s-orbital mark
lineStart2 = 0 # start of higher-orbital marks

electronCount = 0 # counter for electrons as they're placed


#### drawing all the orbitals and electrons ####
for i in range(len(n)):
    elevation_l = 1
    for j in range(l[i]+1):
        for k in range(len(ml[j])):
            lineStart2 = lineStart1 + 1.25*k 
            xmin = lineStart2+k # where each orbital line starts
            xmax = lineStart2+1+k # where the line ends
            y = elevation_n + elevation_l*j # orbital line's elevation
            ax.hlines(y, xmin, xmax, color = "black") # the lines themselves
            ax.hlines(y, xmax, xmax+0.1, color = 'white') # spacing between lines

            
            if electronCount < 54 and j < 3: # drawing the arrows, given the right filling conditions
                ax.annotate('', xy=(0.125+xmin, 1.5+y),
                         xycoords='data',
                         xytext=(0.125+xmin, y-0.25),
                         textcoords='data',
                         arrowprops=dict(arrowstyle= '->', color='blue', lw=1, ls='-'))
                ax.annotate('', xy=(0.125+xmin+0.625, y-0.5),
                         xycoords='data',
                         xytext=(0.125+xmin+0.625, 1.25+y),
                         textcoords='data',
                         arrowprops=dict(arrowstyle= '->', color='red', lw=1, ls='-'))
                electronCount = electronCount + 2
                
        plt.text(xmin+2, y, str(i+1)+orbitals[j], ha='right', va = 'center', ) # labels for the each orbital
    elevation_n = elevation_n + 5


ax.axis('off')
ax.set_title("Filling Orbitals for 54 Electrons")
plt.show() 

