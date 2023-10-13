### importataion des bibliotheques
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def wa(ninc, na, theta_inc):
    return np.lib.scimath.sqrt(na*na-(ninc*np.sin(theta_inc))**2)

def r(r12, r23, w2, d, lmda):
    numerateur1 = (r12 + r23 * np.exp(1j * (2 * np.pi / lmda) * 2 * d * w2))
    denominateur1 = (1 + r12*r23*np.exp(1j * (2 * np.pi / lmda)* 2 * d * w2))
    return numerateur1 / denominateur1

def t(t12, t23, d, lmda, w2, w3, r12, r23):
    numerateur = t12 * t23 * np.exp(1j * (2 * np.pi / lmda) * d * (w2 - w3))
    denominateur = 1 + r12 * r23 * np.exp(1j * (2 * np.pi / lmda) * 2 * d * w2)
    return numerateur / denominateur

nbpt_theta = 1000
nbpt_d = 1000

theta = np.linspace(0, np.pi/2, nbpt_theta)
d = np.linspace(10, 100, nbpt_d)
#%% Méthode 2D

#Je cree 2 tableaux pour utiliser meshgrid
# avec theta est un tableau unidimensionnel qui contient des valeurs d'angle
# et d qui contient des valeurs de l'epaisseur

#Ensuite, je prends ces deux tableaux et crée deux grilles bidimensionnelles
#"Theta" et "D"
#l'utilisation de meshgrid me permet de ne pas avoir de probleme de taille de matrices

Theta, D = np.meshgrid(theta, d) 

#je pose les données du problemes
n1 = 1.5
n2 = 0.24 * (1 + 12.875 * 1j)
n3 = 1
ninc = n1
lmda = 500

#Je calcule wa et la permitivite pour chaques valeurs recherchees
w1 = wa(ninc, n1, Theta)
w2 = wa(ninc, n2, Theta)
w3 = wa(ninc, n3, Theta)
E_1 = n1**2
E_2 = n2**2
E_3 = n3**2

#comme precedemment, je calcule les valeurs des coefficients
r12 = (E_2*w1 - E_1*w2)/(E_2*w1 + E_1*w2)
r23 = (E_3*w2 - E_2*w3)/(E_3*w2 + E_2*w3)
t12 = (np.sqrt(E_1/E_2))*((2*E_2*w1)/(E_2*w1 + E_1*w2))
t23 = (np.sqrt(E_2/E_3))*((2*E_3*w2)/(E_3*w2 + E_2*w3))

#je calcule la valeur absolue de t
abs_t = np.abs(t(t12, t23, D, lmda, w2, w3, r12, r23))

#je mets theta en degree avec un fonction sur numpy
#methode diffrente que pour la partie 1 du tp mais
#ca marche aussi
theta_deg = np.degrees(theta)

#je trace ma courbe avec contourf et une colorbar montrant les valeurs de abs(t)
plt.figure(7)
plt.contourf(theta_deg, d, abs_t, levels=100, cmap='hot')
plt.colorbar(label='Abs(t)')
plt.xlabel('Angle d\'incidence (degrés)')  
plt.ylabel('Épaisseur (d)')
plt.xlim((30, 60))
plt.ylim((0, 100))
plt.title('Absorption en fonction de l\'angle d\'incidence et de l\'épaisseur')
plt.grid()

#%%
#Je recherche la valeur maximale d'abs(t)
max_abs_t = np.max(abs_t)

#Je recherche les indices de la valeur maximale dans le tableau abs_t
max_idx = np.unravel_index(np.argmax(abs_t), abs_t.shape)

#J'extrais l'angle correspondant à partir de theta en utilisant l'indice
max_theta_deg = np.degrees(theta[max_idx[1]])

#J'extrais de meme l'epaisseur correspndant
max_epaisseur = d[max_idx[0]]

# Afficher les valeurs maximales avec 4 chiffres significatifs
print("Maximum de Abs(t) : {:.4f}".format(max_abs_t))
print("Angle correspondant (degrés) : {:.2f}".format(max_theta_deg))
print("Épaisseur correspondante : {:.2f}".format(max_epaisseur))

#%% Méthode 3D

n1 = 1.5
n2 = 0.24 * (1 + 12.875 * 1j)
n3 = 1
ninc = n1

lmda = 500

nbpt_d = 5000
d = np.linspace(10, 100, nbpt_d)  # Épaisseur de la couche d'argent de 10 à 100 nm

theta_deg = np.linspace(0, 90, 100)  # Variation de l'angle d'incidence de 0 à 90 degrés
Theta, D = np.meshgrid(theta_deg, d)

w1 = wa(ninc, n1, np.radians(Theta))
w2 = wa(ninc, n2, np.radians(Theta))
w3 = wa(ninc, n3, np.radians(Theta))

ϵ_1 = n1**2
ϵ_2 = n2**2
ϵ_3 = n3**2

r12 = (ϵ_2*w1 - ϵ_1*w2)/(ϵ_2*w1 + ϵ_1*w2)
r23 = (ϵ_3*w2 - ϵ_2*w3)/(ϵ_3*w2 + ϵ_2*w3)
t12 = (np.sqrt(ϵ_1/ϵ_2))*((2*ϵ_2*w1)/(ϵ_2*w1 + ϵ_1*w2))
t23 = (np.sqrt(ϵ_2/ϵ_3))*((2*ϵ_3*w2)/(ϵ_3*w2 + ϵ_2*w3))

abst = np.abs(t(t12, t23, D, lmda, w2, w3, r12, r23))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(Theta, D, abst, cmap='hot', rstride=1, cstride=1, antialiased=True)

# Ajoutez des lignes de contour pour une meilleure lisibilité
contours = ax.contour(Theta, D, abst, cmap='hot', linewidths=0.5, linestyles="solid", offset=0, zdir='z')
plt.contourf(Theta, D, abst, levels=100, cmap='hot')

# Ajoutez une colorbar
cbar = fig.colorbar(surface)

# Étiquettes des axes et titre
ax.set_xlabel('Angle d\'incidence (degrés)')
ax.set_ylabel('Épaisseur (nm)')
ax.set_zlabel('Abs(t)')

ax.autoscale_view()
plt.show()




