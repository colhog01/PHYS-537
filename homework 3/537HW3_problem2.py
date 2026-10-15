# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 00:53:33 2026

@author: 12242
"""
from matplotlib import pyplot as plt
from itertools import product
from pathlib import Path
import sympy as sy
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W,H,DPI=3600,2400,600
OUT=Path(__file__).resolve().parents[2]/"Chapter4images2"; OUT.mkdir(exist_ok=True)

NAVY=(24,73,137); BLUE=(54,116,184); LBLUE=(166,205,239); ORANGE=(229,137,28)
DORANGE=(145,76,4); PURPLE=(83,35,151); LPURPLE=(198,174,231); RED=(205,51,48)
GREEN=(13,132,103); BLACK=(22,23,28); GRAY=(118,126,140); GRID=(215,220,228)

# drawing graphics and storing in a folder
def ft(path,size): return ImageFont.truetype(path,size)
def ct(draw,xy,s,f,fill=BLACK): draw.text(xy,s,font=f,fill=fill,anchor="mm")

# arrows on the graphs
def arrow(draw,p,q,color,width=12,head=32):
    p,q=np.asarray(p,float),np.asarray(q,float); u=(q-p)/np.linalg.norm(q-p); n=np.array([-u[1],u[0]])
    draw.line((*p,*q),fill=color,width=width); b=q-head*u
    draw.polygon([tuple(q),tuple(b+.55*head*n),tuple(b-.55*head*n)],fill=color)

A,DIST,T,GAMMA = 2.46, 2.46/np.sqrt(3), -2.70, 0.390
EBAR=(0+0)/2; DELTA=0-0

# k-space in-plane nearest-neighbor vectors
d1 = np.array([DIST*0, DIST*1])
d2 = np.array([DIST*np.sqrt(3)/2, -1*DIST/2])
d3 = np.array([-1*DIST*np.sqrt(3)/2, -1*DIST/2])
neighbors = np.asarray([d1, d2, d3])


L = np.array([0, 0])
K = np.array([4*np.pi/(3*A), 0])
M = np.array([np.pi/A, np.pi/(A*np.sqrt(3))])


p_a1, p_b1, p_a2, p_b2, t, F1, F2, a1, b1, a2, b2 = sy.symbols("p_z_a1 p_z_b1 p_z_a2 p_z_b2 t f(k) f*(k) a_1 b_1 a_2 b_2")
y = sy.gamma
p_a1, p_b1, p_a2, p_b2 = sy.latex(p_a1), sy.latex(p_b1), sy.latex(p_a2), sy.latex(p_b2)
labels = [p_a1, p_b1, p_a2, p_b2]
hamiltonian_construction = np.array([[0, t*F1, 0, 0], [t*F2, 0, y, 0], [0, y, 0, t*F1], [0, t*F2, 0, 0]])
df = pd.DataFrame(hamiltonian_construction, index=labels, columns=labels)

print("Hamiltonian with elements labelled with the orbitals that produce them:")
print(df)

print()

def factor(k, dim):
    k = np.asarray(k)
    f = 0
    for i in range(len(neighbors)):
        f = f + np.exp(1j*np.dot(k[...,:dim],neighbors[i]))
    return f

# Hamiltonian for the 2-layer system, includes a single vertical hopping parameter
def H2L(f): 
    H = np.array([[0, T*f, 0, 0],[T*np.conjugate(f), 0, GAMMA, 0],[0, GAMMA, 0, T*f],[0, 0, T*np.conjugate(f), 0]])
    return H

# checks that the Hamiltonian is Hermitian
def hermiticity_check(hamiltonian_construction):
    H = H2L(factor([0,0], 2))
    if np.all(H)==np.all(np.conjugate(H.T)):
        print("This Hamiltonian is Hermitian.")
    else:
        print("This Hamiltonian is not Hermitian.")

hermiticity_check(H2L(factor([0,0],2)))

# calculates the 4 energy eigenvalues of the new system
def energies2D(H):
    eigenvals = np.linalg.eigvalsh(H)
    low = eigenvals[0]
    mlow  = eigenvals[1]
    mhigh = eigenvals[2]
    high = eigenvals[3]
    return low, mlow, mhigh, high

# calculates the 4 eigenstates of the new system
def vectors(H):
    eigenvecs = np.linalg.eigh(H)[1]
    return eigenvecs

print()

print("The expression for the eigenvalues is calculated in the PDF accompanying these programs. Here is a check to make sure they produce the same values as the numerical calculation for the Γ-point:")
l, ml, mh, h = energies2D(H2L(factor([0,0], 2)))
print("Numerical eigenvalues:", l, ml, mh, h)
frac = GAMMA/2
sqrt = np.sqrt(abs(T*factor([0,0], 2))**2 + (frac)**2)
print("Analytical eigenvalues:", -sqrt-frac, -sqrt+frac, sqrt-frac, sqrt+frac)
if np.isclose(np.array([l, ml, mh, h]).all(), np.array([-sqrt-frac, -sqrt+frac, sqrt-frac, sqrt+frac]).all()):
    print("The numerical and analytical eigenvalues match.")
else:
    print("The numerical and analytical eigenvalues don't match.")
print()

# samples n points along the desired path in the Brillouin Zone for plotting
def sample_path(points,n=220):
    ks=[]; xs=[]; ticks=[0.]; total=0.
    for p,q in zip(points[:-1],points[1:]):
        p,q=np.asarray(p,float),np.asarray(q,float); length=np.linalg.norm(q-p)
        for j in range(n):
            t=j/n; ks.append((1-t)*p+t*q); xs.append(total+t*length)
        total+=length; ticks.append(total)
    ks.append(np.asarray(points[-1],float)); xs.append(total)
    return np.asarray(ks),np.asarray(xs),np.asarray(ticks)
def xy(x,y,bounds,box):
    xmin,xmax,ymin,ymax=bounds; l,t,r,b=box
    return l+(x-xmin)/(xmax-xmin)*(r-l), b-(y-ymin)/(ymax-ymin)*(b-t)

def parameter_box(draw,box):
    draw.rounded_rectangle(box,28,fill="white",outline=GRAY,width=5); x=(box[0]+box[2])/2
    ct(draw,(x,box[1]+55),"Harrison parameters",ft(r'C:\Windows\Fonts\Verdana\verdanab.ttf',41),NAVY)
    rows=["epsilon_Li = -5.34 eV","epsilon_Na = -4.96 eV","a = 4.71 angstrom",
          "d = a/2 = 2.36 angstrom","V_ss_sigma(d) = -1.80 eV"]
    for j,row in enumerate(rows): ct(draw,(x,box[1]+120+51*j),row,ft(r'C:\Windows\Fonts\Times New Roman\times.ttf',32))

# plots the bands for 2-layer graphite based on the energy eigenvalues at each sampled point along the path
def band_plot_3d(title,subtitle,x,lo,midlo,midhi,hi,ticks,labels,name,note,
                  parameter_position=(430,410,1190,780)):
    im=Image.new("RGB",(W,H),"white"); dr=ImageDraw.Draw(im)
    ct(dr,(W/2,95),title,ft(r'C:\Windows\Fonts\Verdana\verdanab.ttf',82),NAVY); ct(dr,(W/2,190),subtitle,ft(r'C:\Windows\Fonts\Times New Roman\times.ttf',43))
    
    box=(360,350,3360,1940); ymin=min(lo.min(),0)-.6; ymax=max(hi.max(),0)+.6
    bounds=(x[0],x[-1],ymin,ymax)
    for e in np.linspace(np.floor(ymin),np.ceil(ymax),9):
        p=xy(x[0],e,bounds,box); q=xy(x[-1],e,bounds,box); dr.line((*p,*q),fill=GRID,width=3)
        label="0" if abs(e)<0.5 else f"{e:.0f}"
        dr.text((box[0]-25,p[1]),label,font=ft(r'C:\Windows\Fonts\Verdana\verdana.ttf',34),fill=BLACK,anchor="rm")
    for e,c,label in [(0,LBLUE,"epsilon_Li"),(0,(245,180,105),"epsilon_Na")]:
        p=xy(x[0],e,bounds,box); q=xy(x[-1],e,bounds,box); dr.line((*p,*q),fill=c,width=5)
        dr.text((box[0]+10,p[1]-10),label,font=ft(r'C:\Windows\Fonts\Times New Roman\times.ttf',31),fill=c,anchor="lb")
    tick_font=38 if len(labels)>6 else 48
    for j,(tick,label) in enumerate(zip(ticks,labels)):
        px=xy(tick,ymin,bounds,box)[0]; dr.line((px,box[1],px,box[3]),fill=(165,172,184),width=4)
        offset=58+(j%2)*36 if len(labels)>6 else 68
        ct(dr,(px,box[3]+offset),label,ft(r'C:\Windows\Fonts\Times New Roman\times.ttf',tick_font))
    for vals,c in [(lo,BLUE), (midlo, NAVY), (midhi,DORANGE), (hi,ORANGE)]:
        dr.line([xy(a,b,bounds,box) for a,b in zip(x,vals)],fill=c,width=11,joint="curve")
    dr.line((box[0],box[1],box[0],box[3]),fill=BLACK,width=8); dr.line((box[0],box[3],box[2],box[3]),fill=BLACK,
            width=8)
    dr.text((150,(box[1]+box[3])/2),"Energy (eV)",font=ft(r'C:\Windows\Fonts\Verdana\verdana.ttf',47),fill=BLACK,anchor="mm")
    im.save(OUT/name,dpi=(DPI,DPI))

# calculates the bands for the 2-layer graphite along the path Γ, K, M, Γ
def bands_3d():
    ks,s,t=sample_path([L, K, M, L],260)
    lo = np.array([])
    midlo = np.array([])
    midhi = np.array([])
    hi = np.array([])
    for i in range(len(ks)):
        under1, under2, over1, over2 = energies2D(H2L(factor(ks[i],2)))
        lo = np.append(lo, under1)
        midlo = np.append(midlo, under2)
        midhi = np.append(midhi, over1)
        hi = np.append(hi, over2)
    band_plot_3d("2D graphite bands","Path Γ-K-M-Γ through the diamond-shaped Brillouin zone",
        s,lo,midlo,midhi,hi,t,["Γ","K","M","Γ"],"graphite-2layers-2d-high-symmetry-bands-harrison-600dpi.png",
        "The flat X-M edge is exact only for the nearest-neighbor checkerboard model.",
        parameter_position=(1360,390,2120,760))

# Calculating bands near the K-point
def bands_3d_K():
   ks,s,t=sample_path([[K[0]-0.1,K[1]], K, [K[0]+0.1, K[1]]],260)
   lo = np.array([])
   midlo = np.array([])
   midhi = np.array([])
   hi = np.array([])
   for i in range(len(ks)):
       under1, under2, over1, over2 = energies2D(H2L(factor(ks[i],2)))
       lo = np.append(lo, under1)
       midlo = np.append(midlo, under2)
       midhi = np.append(midhi, over1)
       hi = np.append(hi, over2)
   band_plot_3d("2D graphite bands","Path around K through the diamond-shaped Brillouin zone",
       s,lo,midlo,midhi,hi,t,["K-","K","K+"],"graphite-2layers-2d-K-bands-harrison-600dpi.png",
       "The flat X-M edge is exact only for the nearest-neighbor checkerboard model.",
       parameter_position=(1360,390,2120,760))

### Reporting the eigenvalues and weights at K, checking that the energies match the expected values ###
weights = np.zeros((4, 4))
for i in range(4):
    l = vectors(H2L(factor(K, 2)))[:,i]
    for j in range(len(l)):
        if np.isclose(np.abs(l[j])**2, 0):
            weights[i][j] = 0
        else:
            weights[i][j]=np.abs(l[j])**2
K_energies = np.asarray(energies2D(H2L(factor(K, 2))))
for i in range(4):
    if np.isclose(K_energies[i], 0):
        K_energies[i] = 0
K_expected = np.array([-GAMMA, 0, 0, GAMMA])
print("The four eigenvalues at the K-point are: ")
print(K_energies)
print("Compared to the expected eigenvalues:")
print(K_expected, ",")

if K_expected.all() == K_energies.all():
    print("The model predicts the expected energy values at the K-point.")
else:
    print("The model does not predict the expected energy values at the K-point.")

print()
print("The weights of the eigenvectors at the K-point are:")
print("Eigenstate 1: ", weights[0])
print("Eigenstate 2: ", weights[1])
print("Eigenstate 3: ", weights[2])
print("Eigenstate 4: ", weights[3])
print()

# Row and column labels
rows = ["A1", "B1", "A2", "B2"]
cols = ["Band 1", "Band 2", "Band 3", "Band 4"]

fig, ax = plt.subplots(figsize=(6, 4))

# Creates the heatmap
im = ax.imshow(weights, cmap="winter")

ax.set_xticks(range(len(cols)), labels=cols, rotation=45, ha="right")
ax.set_yticks(range(len(rows)), labels=rows)

# Annotating the cells with weights
for i in range(len(rows)):
    for j in range(len(cols)):
        ax.text(j, i, f"{weights[i, j]:.1f}", ha="center", va="center", color="magenta")

fig.colorbar(im, ax=ax, label="weight")
ax.set_title("Weights at K")
plt.tight_layout()

# saving heat map as 600-DPI png PIL object
plt.savefig(OUT/"graphene-2layers-heatmap.png",format='png',dpi=DPI)

### Describing the eigenvectors as dimer or non-dimer
eigenvecs = vectors(H2L(factor(K, 2)))
for i in range(len(eigenvecs)):
    for j in range(len(eigenvecs[:,i])):
        if np.isclose(eigenvecs[i][j],0):
            eigenvecs[i][j] = 0
print("At the K-point:")
for i in range(len(vectors(H2L(factor(K, 2))))):
    print("The vector corresponding to Band "  + str(i+1) + " is")
    print(eigenvecs[:,i])
    if np.isclose(vectors(H2L(factor(K, 2)))[1,i], 0) == False and np.isclose(vectors(H2L(factor(K, 2)))[2,i], 0) == False:
        print("and it is a dimer site.")
    else:
        print("and it is a non-dimer site.")
    print()

bands_3d(), bands_3d_K()