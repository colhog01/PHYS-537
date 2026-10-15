# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:46:09 2026

@author: 12242
"""
import sympy as sy
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W,H,DPI=3600,2400,600 # resolution of figures
OUT=Path(__file__).resolve().parents[2]/"Chapter4images2"; OUT.mkdir(exist_ok=True) # folder to hold saved figures

# color coordinates in RGB
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

# constants: 
    # A = in-plane lattice constant, angstrom
    # DIST = nearest neighbor C-C distance, angstrom
    # C = height of the two-layer Bernal repeat (C/2 from one layer to the next), angstrom
    # T = nearest-neighbor in-plane hopping, eV
    # GAMMA = vertical hopping of the AB dimer pair
A,DIST,T,GAMMA,C = 2.46, 2.46/np.sqrt(3), -2.70, 0.390, 6.70


# k-space in-plane nearest-neighbor vectors
d1 = np.array([DIST*0, DIST*1])
d2 = np.array([DIST*np.sqrt(3)/2, -1*DIST/2])
d3 = np.array([-1*DIST*np.sqrt(3)/2, -1*DIST/2])
neighbors = np.asarray([d1, d2, d3])

# high-symmetry points
L = np.array([0, 0, 0]) # gamma point
K = np.array([4*np.pi/(3*A), 0, 0]) # K point
M = np.array([np.pi/A, np.pi/(A*np.sqrt(3)), 0]) # M point
A2 = np.array([0, 0, np.pi/C]) # A point
H2 = np.array([4*np.pi/(3*A), 0, np.pi/C]) # H point
L2 = np.array([np.pi/A, np.pi/(A*np.sqrt(3)), np.pi/C]) # L point

def factor(k, dim):
    k = np.asarray(k)
    f = 0
    for i in range(len(neighbors)):
        f = f + np.exp(1j*np.dot(k[...,:dim],neighbors[i]))
    return f

# new vertical hopping sum, different than the single parameter in the 2-layer case since electrons can hop up and down layers
# based on the k-space z-coordinate of the same vector used in factor
def gactor(k): 
    g = 2*GAMMA*np.cos(k*C/2)
    return g

# Hamiltonian based on both hopping sums
def H3D(f,g): 
    H = np.array([[0, T*f, 0, 0],[T*np.conjugate(f), 0, g, 0],[0, g, 0, T*f],[0, 0, T*np.conjugate(f), 0]])
    return H

def energies2D(H):
    eigenvals = np.linalg.eigvalsh(H)
    low = eigenvals[0]
    mlow  = eigenvals[1]
    mhigh = eigenvals[2]
    high = eigenvals[3]
    return low, mlow, mhigh, high

def vectors(H):
    eigenvecs = np.linalg.eigh(H)[1]
    return eigenvecs

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

# plots the bands for 3D graphite based on the energy eigenvalues at each sampled point along the path
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

# calculates the 3D graphite bands
def bands_3d(): 
    ks,s,t=sample_path([L, K, M, L, A2, H2, L2, A2],260) # using sample_path to sample the path for energy values
    
    # arrays to store the lowest, second lowest, second highest, and highest eigenenergies respectively at every point
    lo = np.array([])
    midlo = np.array([])
    midhi = np.array([])
    hi = np.array([])
    for i in ks: # pulling all the energies
        under1, under2, over1, over2 = energies2D(H3D(factor(i[:2],2),gactor(i[2])))
        lo = np.append(lo, under1)
        midlo = np.append(midlo, under2)
        midhi = np.append(midhi, over1)
        hi = np.append(hi, over2)
    
    # plotting the bands with band_plot_3d using the energies at the sampled points along the desired path
    band_plot_3d("3D graphite bands","Path Γ-K-M-Γ-A-H-L-A through the layered hexagonal Brillouin zone",
        s,lo,midlo,midhi,hi,t,["Γ","K","M","Γ","A","H","L","A"],"graphite-3d-high-symmetry-bands-harrison-600dpi.png",
        "",
        parameter_position=(1360,390,2120,760))



# verifying the limiting forms at K and H, 
# expect K (kz = 0) energies to be (-2*GAMMA, 0, 0, 2*GAMMA) and H (g = 0) energies to all be zero
K_check = np.asarray(energies2D(H3D(factor(K[:2],2),gactor(K[2]))))
H_check = np.asarray(energies2D(H3D(factor(H2[:2],2),gactor(H2[2]))))

for i in range(len(K_check)):
    if np.isclose(K_check[i], 0):
        K_check[i] = 0
    if np.isclose(H_check[i], 0):
        H_check[i] = 0
print("Energies at K:", K_check)
print("Energies at H:", H_check)

# expected energies at K and H
K_true = np.array([-2*GAMMA, 0, 0, 2*GAMMA])
H_true = np.array([0, 0, 0, 0])

print("Expected energies at K:", K_true)
print("Expected energies at H:", H_true)

# counters to check that the expected and calculated values are all the same
k = 0
h = 0
for i in range(4):
    if np.isclose(K_check[i], K_true[i]):
        k += 1
    if np.isclose(H_check[i], H_true[i]):
        h += 1
if k == h:
    print("Limiting forms at K and H verified.")

def reciprocal_vectors(): # calculates the reciprocal lattice vectors for the honeycomb lattice
    # primitive lattice vector components scale by A as opposed to those of the diamond silicon which scaled by A/2.
    # this is because the lattice vectors in the diamond point from a corner to the center of a face in the conventional cubic cell,
    # while the primitive lattice vectors in graphite point to equivalent sites in an adjacent hexagon.
    # A is used for in-plane vectors, while C is used for the vector pointing between layers.
    # the cos and sin are here because the origin is taken to be a site on a hexagon, so one equivalent site is pi/6 rad rotated above
    # the positive horizontal axis (which contains an inequivalent site), and a second equivalent site will be at a rotation
    # of pi/6 rad below the positive horizontal axis. Then these vectors collect two halves of one type of site and one whole of
    # the other type of site.
    
    # real-space lattice vectors
    a1 = [A*np.cos(np.pi/6), A*np.sin(np.pi/6), 0]
    a2 = [A*np.cos(-np.pi/6), A*np.sin(-np.pi/6), 0]
    a3 = [0, 0, C*1]
    
    # lattice volume
    volume = np.dot(a1,np.cross(a2,a3))
    
    # k-space lattice vectors
    b1 = 2.0*np.pi*np.cross(a2,a3)/volume
    b2 = 2.0*np.pi*np.cross(a3,a1)/volume
    b3 = 2.0*np.pi*np.cross(a1,a2)/volume
    
    return b1,b2,b3

#### functions for calculating and plotting 3D Bernal graphite's density of states ####
def gaussian_kernel(bin_width,sigma):
    half_width=int(np.ceil(5.0*sigma/bin_width))
    offsets=np.arange(-half_width, half_width+1)*bin_width
    kernel=np.exp(-0.5*(offsets/sigma)**2)
    return kernel/kernel.sum()

def smooth_histogram(values,weights,edges,sigma):
    counts,_=np.histogram(values,bins=edges, weights=weights)
    bin_width=edges[1]-edges[0]
    return np.convolve(counts,gaussian_kernel(bin_width, sigma), mode="same")/bin_width

def calculate_dos(mesh_size=36,number_of_bins=1600, sigma=0.075):
    ###Sample one primitive reciprocal cell and return total and projected DOS
    
    b1,b2,b3= reciprocal_vectors()
    fractional=np.arange(mesh_size,dtype=float)/mesh_size-0.5
    number_of_k_points = mesh_size**3
    eigenvalues=np.empty((number_of_k_points,4),dtype=float)
    point=0
    for u1 in fractional:
        for u2 in fractional:
            for u3 in fractional:
                k=u1*b1+u2*b2+u3*b3            
                values=np.linalg.eigvalsh(H3D(factor([k[0],k[1]],2),gactor(k[2])))
                eigenvalues[point] = values
                point+=1    
    
    flat_energies=eigenvalues.ravel()
    edges=np.linspace(flat_energies.min()-0.75,flat_energies.max()+0.75,
                      number_of_bins+1)
    energy=0.5* (edges[:-1]+ edges[1:])
    normalization=1.0/number_of_k_points # excluding spin
    total=smooth_histogram(flat_energies, np.full(flat_energies.size,normalization),
                           edges,sigma)
    return energy,total

# build the DOS plot
def DOS_plot():
    
    NAVY="#17488b"
    BLUE="#2f6fb2"
    GRAY="#6f7784"
    GRID="#d9dee7"
    
    mesh_size=36
    sigma=0.075
    energy,total=calculate_dos(
        mesh_size=mesh_size,sigma=sigma
    )

    fig=plt.figure(figsize=(14.0, 8.4), dpi=300, facecolor="white")
    fig.suptitle(
        r"3D Bernal Graphite: DOS in the $(p_z,p_z)$ basis, spin excluded",
        fontsize=21,
        color=NAVY,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.892,
        rf"Full 3D reciprocal-cellmesh: ${mesh_size}^3$ points; Gaussian broadening $\sigma = {sigma:.3f}$eV; Normalization: 1/${mesh_size}^3$",
        ha="center",
        fontsize=12.5,
        color=GRAY,
    )
    
    ax=fig.add_axes([0.075,0.14,0.69,0.68])
    ax.fill_between(energy,total,color="#d9dee7",alpha=0.65, label="Total DOS")
    ax.plot(energy, total, color="#22252a",linewidth=2.0)
    ax.axvline(0.0, color=BLUE,linestyle="--",linewidth=1.4)
    ax.set_xlabel(r"$Energy $(eV)",fontsize=13)
    ax.set_ylabel("DOS (states/eV/primitive cell)",fontsize=13)
    ax.set_xlim(energy.min(),energy.max())
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="both",color=GRID,linewidth=0.7)
    ax.legend(loc="upper right",frameon=True, framealpha=0.96, fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    info=fig.add_axes([0.79,0.16,0.19,0.64])
    info.set_xlim(0.0, 1.0)
    info.set_ylim(0.0, 1.0)
    info.axis("off")
    info.text(0.5, 0.97,"State counting", ha="center",va="top",
              fontsize=14.5,color=NAVY,fontweight="bold")
    info.text(0.04, 0.83, r"$\int D(E)\, dE$", fontsize=12,color=GRAY)
    info.text(0.96, 0.83, "4 states/cell", ha="right",fontsize=12,
              color="#22252a",fontweight="bold")
    info.text(0.04, 0.73, "Vertical hopping coefficient:",fontsize=12,color=GRAY)
    info.text(0.96, 0.67, rf"{GAMMA} eV",ha="right",fontsize=12,
              color=BLUE,fontweight="bold")
    info.plot([0.04, 0.96], [0.64, 0.64], color=GRID,linewidth=1.2)
    
    output=OUT/ "graphite-density-of-states-600dpi.png"
    fig.savefig(output,dpi=600)
    print(f"saved: {output}")
    
    bin_width=energy[1]-energy[0]
    print(f"integrated total DOS = {np.sum(total) *bin_width:.6f} states/cell")


print()

# estimating the vertical pz-pz sigma hopping parameter with Harrison's rule
h2me = 7.62 # eV angstrom^2
dp = 3.35 # angstrom
Vppo = 2.22*h2me/(dp**2)

print("Estimated Harrison pz-pz hopping parameter Vppo =", Vppo, "eV")
print("Compared to gamma1 =", 0.390, "eV")
print()
if Vppo > 0.39:
    print("The Harrison parameter is about", round(Vppo/0.39, 6), "times larger than the experimental parameter.")
elif Vppo < 0.39:
    print("The experimental parameter is about", round(0.39/Vppo, 6), "times larger than the Harrison parameter.")
else:
    print("The Harrison parameter is the same as the experimental parameter.")
print()

#### plotting the bands and DOS for 3D bernal graphite with the gamma sum replaced by the Harrison parameter ####
#### it's literally all the same as above except that the hamiltonian was slightly modified for the new parameter ####
def bands_3d_harrison():
    ks,s,t=sample_path([L, K, M, L, A2, H2, L2, A2],260)
    lo = np.array([])
    midlo = np.array([])
    midhi = np.array([])
    hi = np.array([])
    for i in ks:
        under1, under2, over1, over2 = energies2D(H3D_harrison(factor(i[:2],2), Harrison_gactor(i[2])))
        lo = np.append(lo, under1)
        midlo = np.append(midlo, under2)
        midhi = np.append(midhi, over1)
        hi = np.append(hi, over2)
    band_plot_3d("2D graphite bands with Harrison parameter","Path Γ-K-M-Γ through the Brillouin zone",
        s,lo,midlo,midhi,hi,t,["Γ","K","M","Γ","A","H","L","A"],"graphite-3d-high-symmetry-bands-harrison-parameter-600dpi.png",
        "The flat X-M edge is exact only for the nearest-neighbor checkerboard model.",
        parameter_position=(1360,390,2120,760))

def calculate_dos_harrison(mesh_size=36,number_of_bins=1600, sigma=0.075):
    ### Sample one primitive reciprocal cell and return total and projected DOS
    
    b1,b2,b3= reciprocal_vectors()
    fractional=np.arange(mesh_size,dtype=float)/mesh_size-0.5
    number_of_k_points = mesh_size**3
    eigenvalues=np.empty((number_of_k_points,4),dtype=float)
    point=0
    for u1 in fractional:
        for u2 in fractional:
            for u3 in fractional:
                k = u1*b1+u2*b2+u3*b3
                values=np.linalg.eigvalsh(H3D_harrison(factor([k[0],k[1]],2), Harrison_gactor(k[2])))
                eigenvalues[point] = values
                point+=1
    
    flat_energies=eigenvalues.ravel()    
    edges=np.linspace(flat_energies.min()-0.75,flat_energies.max()+0.75,
                      number_of_bins+1)
    energy=0.5* (edges[:-1]+ edges[1:])
    normalization=1.0/number_of_k_points # excluding spin
    total=smooth_histogram(
        flat_energies, np.full(flat_energies.size,normalization),edges,sigma
    )
    return energy,total

def DOS_plot_harrison():
    
    NAVY="#17488b"
    BLUE="#2f6fb2"
    GRAY="#6f7784"
    GRID="#d9dee7"
    
    mesh_size=36
    sigma=0.075
    energy,total=calculate_dos_harrison(
        mesh_size=mesh_size,sigma=sigma
    )

    fig=plt.figure(figsize=(14.0, 8.4), dpi=300, facecolor="white")
    fig.suptitle(
        r"3D Bernal Graphite: DOS in the $(p_z,p_z)$ basis, spin excluded, with Harrison parameter",
        fontsize=21,
        color=NAVY,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.892,
        rf"Full 3D reciprocal-cellmesh: ${mesh_size}^3$ points; Gaussian broadening $\sigma = {sigma:.3f}$eV; Normalization: 1/${mesh_size}^3$",
        ha="center",
        fontsize=12.5,
        color=GRAY,
    )
    ax=fig.add_axes([0.075,0.14,0.69,0.68])
    ax.fill_between(energy,total,color="#d9dee7",alpha=0.65, label="Total DOS")
    ax.plot(energy, total, color="#22252a",linewidth=2.0)
    ax.axvline(0.0, color=BLUE,linestyle="--",linewidth=1.4)
    ax.set_xlabel(r"$Energy $(eV)",fontsize=13)
    ax.set_ylabel("DOS (states/eV/primitive cell)",fontsize=13)
    ax.set_xlim(energy.min(),energy.max())
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="both",color=GRID,linewidth=0.7)
    ax.legend(loc="upper right",frameon=True, framealpha=0.96, fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    info=fig.add_axes([0.79,0.16,0.19,0.64])
    info.set_xlim(0.0, 1.0)
    info.set_ylim(0.0, 1.0)
    info.axis("off")
    info.text(0.5, 0.97,"State counting", ha="center",va="top",
              fontsize=14.5,color=NAVY,fontweight="bold")
    info.text(0.04, 0.83, r"$\int D(E)\, dE$", fontsize=12,color=GRAY)
    info.text(0.96, 0.83, "4 states/cell", ha="right",fontsize=12,
              color="#22252a",fontweight="bold")
    info.text(0.04, 0.73, "Harrison parameter:",fontsize=12,color=GRAY)
    info.text(0.96, 0.67, rf"{round(Vppo,4)} eV",ha="right",fontsize=12,
               color=BLUE,fontweight="bold")
    info.plot([0.04, 0.96], [0.64, 0.64], color=GRID,linewidth=1.2)
    
    output=OUT/ "graphite-density-of-states-harrison-600dpi.png"
    fig.savefig(output,dpi=600)
    print(f"saved: {output}")
    
    bin_width=energy[1]-energy[0]
    print(f"integrated total DOS = {np.sum(total) *bin_width:.6f} states/cell")

def Harrison_gactor(k): # k-space inter-layer hopping sum
    g = 2*Vppo*np.cos(k*C/2)
    return g

# the same structure as the hamiltonian before except the references to g are replaced with the calculated Harrison parameter
def H3D_harrison(f, g):
    H = np.array([[0, T*f, 0, 0],[T*np.conjugate(f), 0, g, 0],[0, g, 0, T*f],[0, 0, T*np.conjugate(f), 0]])
    return H


bands_3d(), DOS_plot(), bands_3d_harrison(), DOS_plot_harrison()