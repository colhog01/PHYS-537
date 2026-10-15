# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:39:32 2026

@author: 12242
"""
from itertools import product
from pathlib import Path
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

ELI,ENA,A,DIST,T,GAMMA = 0, 0, 2.46, 2.46/np.sqrt(3), -2.70, 0.390
EBAR=(ELI+ENA)/2; DELTA=ELI-ENA

# k-space in-plane nearest-neighbor vectors
d1 = np.array([DIST*0, DIST*1])
d2 = np.array([DIST*np.sqrt(3)/2, -1*DIST/2])
d3 = np.array([-1*DIST*np.sqrt(3)/2, -1*DIST/2])
neighbors = np.asarray([d1, d2, d3])

# the k-space hopping sum based on k-vector and its dimension; since it's a single layer, hopping is within the layer
def factor(k, dim):
    k = np.asarray(k)
    f = 0
    for i in range(len(neighbors)):
        f = f + np.exp(1j*np.dot(k[...,:dim],neighbors[i]))
    return f

# single-layer Hamiltonian based on the hopping sum
def H1L(f):
    H = np.array([[0, T*f],[T*np.conjugate(f), 0]])
    return H

# calculates energy eigenvalues for each Hamiltonian
def energies2D(H):
    eigenvals = np.linalg.eigvalsh(H)
    low = eigenvals[0]
    high = eigenvals[1]
    return low, high

L = np.array([0, 0])
K = np.array([4*np.pi/(3*A), 0])
M = np.array([np.pi/A, np.pi/(A*np.sqrt(3))])
print("|f| for each symmetry point is")
print("L", abs(factor(L,2)))
print("K", abs(factor(K,2)))
print("M", abs(factor(M,2)))
print()
print("The energies at each symmetry point are")
print("L", energies2D(H1L(factor(L,2))))
print("K", energies2D(H1L(factor(K,2))))
print("M", energies2D(H1L(factor(M,2))))

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

# plots the bands for single-layer graphite based on the energy eigenvalues at each sampled point along the path
def band_plot(title,subtitle,x,lo,hi,ticks,labels,name,note,
                  parameter_position=(430,410,1190,780)):
    im=Image.new("RGB",(W,H),"white"); dr=ImageDraw.Draw(im)
    ct(dr,(W/2,95),title,ft(r'C:\Windows\Fonts\Verdana\verdanab.ttf',82),NAVY); ct(dr,(W/2,190),subtitle,ft(r'C:\Windows\Fonts\Times New Roman\times.ttf',43))
    
    box=(360,350,3360,1940); ymin=min(lo.min(),ELI)-.6; ymax=max(hi.max(),ENA)+.6
    bounds=(x[0],x[-1],ymin,ymax)
    for e in np.linspace(np.floor(ymin),np.ceil(ymax),9):
        p=xy(x[0],e,bounds,box); q=xy(x[-1],e,bounds,box); dr.line((*p,*q),fill=GRID,width=3)
        label="0" if abs(e)<0.5 else f"{e:.0f}"
        dr.text((box[0]-25,p[1]),label,font=ft(r'C:\Windows\Fonts\Verdana\verdana.ttf',34),fill=BLACK,anchor="rm")
    for e,c,label in [(ELI,LBLUE,"epsilon_Li"),(ENA,(245,180,105),"epsilon_Na")]:
        p=xy(x[0],e,bounds,box); q=xy(x[-1],e,bounds,box); dr.line((*p,*q),fill=c,width=5)
        dr.text((box[0]+10,p[1]-10),label,font=ft(r'C:\Windows\Fonts\Times New Roman\times.ttf',31),fill=c,anchor="lb")
    tick_font=38 if len(labels)>6 else 48
    for j,(tick,label) in enumerate(zip(ticks,labels)):
        px=xy(tick,ymin,bounds,box)[0]; dr.line((px,box[1],px,box[3]),fill=(165,172,184),width=4)
        offset=58+(j%2)*36 if len(labels)>6 else 68
        ct(dr,(px,box[3]+offset),label,ft(r'C:\Windows\Fonts\Times New Roman\times.ttf',tick_font))
    for vals,c in [(lo,NAVY),(hi,ORANGE)]:
        dr.line([xy(a,b,bounds,box) for a,b in zip(x,vals)],fill=c,width=11,joint="curve")
    dr.line((box[0],box[1],box[0],box[3]),fill=BLACK,width=8); dr.line((box[0],box[3],box[2],box[3]),fill=BLACK,
            width=8)
    dr.text((150,(box[1]+box[3])/2),"Energy (eV)",font=ft(r'C:\Windows\Fonts\Verdana\verdana.ttf',47),fill=BLACK,anchor="mm")
   #parameter_box(dr,parameter_position); ct(dr,(W/2,2170),note,ft(r'C:\Windows\Fonts\Verdana\verdanab.ttf',41),PURPLE)
    im.save(OUT/name,dpi=(DPI,DPI))


### 2D plot of the bands over a hexagonal lattice

def surface_projection_graphite(kx, ky, e):
    return np.array([1800 + 1100*kx + 300*ky, 
                     1150 + 160*ky - 105*e]) 

def surfaces_2d_graphite():
    im=Image.new("RGB",(W,H),"white"); dr=ImageDraw.Draw(im)
    ct(dr,(W/2,95),"2D Single-Layer Graphite Bands over the First Brillouin Zone",ft(r'C:\Windows\Fonts\Verdana\verdanab.ttf',80),NAVY)
    ct(dr,(W/2,185),"E_±(k_x,k_y)",ft(r'C:\Windows\Fonts\Times New Roman\times.ttf',43))
    
    # Defining a range that extends just beyond the Brillouin Zone and making a mesh
    xrange = 4*np.pi/(3*A) 
    yrange = np.pi/(np.sqrt(3)*A) 

    kx_grid = np.linspace(-xrange * 1.1, xrange * 1.1, 50) 
    ky_grid = np.linspace(-yrange * 1.1, yrange * 1.1, 50)

    overlay=Image.new("RGBA",(W,H),(255,255,255,0)); od=ImageDraw.Draw(overlay,"RGBA")

    for branch,base_color,edge_color in [(0,(48,91,160),NAVY),(1,(231,139,48),DORANGE)]: # Lower and Upper bands
        cells=[]
        for i in range(len(kx_grid)-1):
            for j in range(len(ky_grid)-1):
                k_points = [(kx_grid[i], ky_grid[j]),(kx_grid[i+1], ky_grid[j]),
                           (kx_grid[i+1], ky_grid[j+1]),(kx_grid[i], ky_grid[j+1])]
                polygon=[]; depth=0.
                for qx, qy in k_points:
                    k_vec = np.array([qx,qy]); depth += qy
                    kx_norm = qx/xrange
                    ky_norm = qy/yrange
                    lo, hi = energies2D(H1L(factor(k_vec, 2)))
                    polygon.append(tuple(surface_projection_graphite(kx_norm, ky_norm, (lo, hi)[branch])))
                cells.append((depth/len(k_points), polygon)) # Average ky for sorting

        for depth,polygon in sorted(cells,key=lambda item:item[0]):
            shade=max(-30,min(30,int(22 * (depth / yrange))))
            color=tuple(max(0,min(255,c+shade)) for c in base_color)+(145,)
            od.polygon(polygon,fill=color,outline=edge_color+(90,))
    
    im=Image.alpha_composite(im.convert("RGBA"),overlay).convert("RGB"); dr=ImageDraw.Draw(im)
    
    # boundary is now a hexagon instead of a rectangle
    boundary=[(-1,0),(-0.5,-np.sqrt(3)/2),(0.5,-np.sqrt(3)/2),(1,0),(0.5,np.sqrt(3)/2),(-0.5,np.sqrt(3)/2),(-1,0)]
    dr.line([tuple(surface_projection_graphite(x,y,ELI)) for x,y in boundary],fill=DORANGE,width=11,joint="curve")
    ct(dr,(610,780),"upper band",ft(r'C:\Windows\Fonts\Verdana\verdanab.ttf',45),DORANGE)
    ct(dr,(2960,1740),"lower band",ft(r'C:\Windows\Fonts\Verdana\verdanab.ttf',45),NAVY)
    dr.rounded_rectangle((520,2115,3080,2290),24,fill="white",outline=LPURPLE,width=4)
    ct(dr,(W/2,2200),"Dirac points are where the bands touch the Fermi-level: all K-points.",ft(r'C:\Windows\Fonts\Verdana\verdanab.ttf',41),
       PURPLE)
    im.save(OUT/"graphite-2d-band-surfaces.png",dpi=(DPI,DPI))


# calculating the bands along the path Γ, K, M, Γ
def bands_2d():
    ks,s,t=sample_path([L, K, M, L],260)
    lo = np.array([])
    hi = np.array([])
    for i in range(len(ks)):
        under, over = energies2D(H1L(factor(ks[i],2)))
        lo = np.append(lo, under)
        hi = np.append(hi, over)
    band_plot("2D graphite bands","Path Γ-K-M-Γ through the diamond-shaped Brillouin zone",
        s,lo,hi,t,["Γ","K","M","Γ"],"graphite-2d-high-symmetry-bands-harrison-600dpi.png",
        "The flat X-M edge is exact only for the nearest-neighbor checkerboard model.",
        parameter_position=(1360,390,2120,760))

# Calculating bands near the K-point
def bands_2d_K():
   ks,s,t=sample_path([[K[0]-0.1,K[1]], K, [K[0]+0.1, K[1]]],260)
   lo = np.array([])
   hi = np.array([])
   for i in range(len(ks)):
       under, over = energies2D(H1L(factor(ks[i],2)))
       lo = np.append(lo, under)
       hi = np.append(hi, over)
   band_plot("2D graphite bands","Path around K through the hexagonal Brillouin zone",
       s,lo,hi,t,["K-","K","K+"],"graphite-2d-K-bands-harrison-600dpi.png",
       "",
       parameter_position=(1360,390,2120,760))
   qt = np.where((hi==energies2D(H1L(factor(K,2)))[1]))
   fermi = abs((hi[qt[0][0]+1]-hi[qt[0][0]])/(s[qt[0][0]+1]-s[qt[0][0]]))
   hbar = 6.582119569*10**(-16)
   angstrom = 10**(-10)
   print("Numerical Energy rate: ", fermi, " eV")
   print("Expected Energy rate: ", 1.5*abs(T)*DIST, " eV")
   print()
   print("Numerical Fermi velocity: ", fermi*angstrom/hbar, " m/s")
   print("Expected Fermi velocity: ", 1.5*abs(T)*DIST*angstrom/hbar, " m/s")
   
   

bands_2d(), surfaces_2d_graphite(), bands_2d_K()
