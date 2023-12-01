# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:12:25 2023

@author: seghir_c
"""
import numpy as np
import matplotlib.pyplot as plt


Lx = 10
Ly = 10

dt = 1e-2
dx = 1
dy = 1
a = 4

Max_T = 200
T_global = np.zeros((Max_T, Ly, Lx))


Matrice_coeffs = np.ones((Ly, Lx))*2


Matrice_coeffs[2:4, 1:3] = 10
Matrice_coeffs[8:10, 1:3] = 0.1

Matrice_temp_0 = np.zeros((Ly, Lx))

def afficher_grille_temperature(temperature, m):
    #plt.imshow(temperature, cmap='hot', interpolation='nearest',  vmin=0, vmax=4)
    plt.contourf(temperature, cmap='hot', levels=50)
    plt.colorbar(label='Température')
    plt.title(f'Évolution de la température t = {m}dt')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.show()

def prochain_instant(Mat_temp, Mat_coef, dt):
    Mat_temp_resultat = np.zeros_like(Mat_temp)
    for i in range(Ly):
        for j in range(Lx):
            A = Mat_temp[i][j]
            B = 0
            C = 0
            D = 0
            E = 0
            if j == 0 and i > Ly/2 - a and i < Ly/2 + a:
                E = 10
                B = Mat_temp[i - 1][j]
                c = Mat_temp[i][j + 1]
                D = Mat_temp[i + 1][j]
                Mat_temp_resultat[i][j] = Mat_coef[i][j]*dt*((C-2*A + E)/(dx**2)+(D - 2*A + B)/(dy**2)) + A
            elif j == Lx - 1:
                Mat_temp_resultat[i][j]
            elif i ==  0:
                Mat_temp_resultat[i][j]
            elif i == Ly - 1:
                Mat_temp_resultat[i][j]
            else:
                B = Mat_temp[i - 1][j]
                C = Mat_temp[i][j + 1]
                D = Mat_temp[i + 1][j]
                E = Mat_temp[i][j - 1]
                
                Mat_temp_resultat[i][j] = Mat_coef[i][j]*dt*((C-2*A + E)/(dx**2)+(D - 2*A + B)/(dy**2)) + A
                
    return Mat_temp_resultat

t = 0
T_global[0] = Matrice_temp_0


while t < Max_T - 1:
    T_global[t + 1] = prochain_instant(T_global[t], Matrice_coeffs, dt)
    t += 1


    
afficher_grille_temperature(T_global[1], 1)
afficher_grille_temperature(T_global[15], 15)
afficher_grille_temperature(T_global[60], 60)
afficher_grille_temperature(T_global[150], 150)

plt.imshow(np.log(Matrice_coeffs), cmap='viridis')
plt.colorbar(label='D')
plt.title(f'Matrice de coefficient de diffusion')
plt.gca().invert_yaxis()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



#Matrice_coeffs[2:4, 1:3] = 15
#Matrice_coeffs[8:10, 1:3] = 0

# 
