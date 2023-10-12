import numpy as np
import matplotlib.pyplot as plt #importation des bibliothèques numpy et matplotlib.pyplot

#%% Définition de la fonction wa et de theta

def wa(ninc, na, theta_inc):
    return np.lib.scimath.sqrt(na*na-(ninc*np.sin(theta_inc))**2) #définition de la fonction wa

nbpt = 100

theta = np.linspace(0, np.pi/2, nbpt) #on choisit un certain nombre de theta, ici 100, compris entre 
                                      #entre 0 et 2*pi

theta_deg = theta * 180 / np.pi #cela nous permet d'é©crire theta en degrÃ© 

#%% TE Cas 1 : Calcul et tracé des coefficients r,t,R,T pour TE pour 1 dioptre

### Cas 1 
    
n1 = 1.0
n2 = 1.5
ninc = n1 #nous définissons les indices n1 et n2 qui seront Ã  faire varier pour le deuxième cas

w1 = wa(ninc, n1, theta)
w2 = wa(ninc, n2, theta) 

# r polarisation TE 

r12 = (w1 - w2)/(w1 + w2) #permet de calculer le coefficients de réflexion pour la polarisation TE

plt.figure(1)
plt.plot(theta_deg, r12, label='r', color='red') #nous traaçons r12 = f(theta_deg)
plt.ylim((-1, -0.2))
plt.xlabel('Angle d incidence') #permet de nommer les axes
plt.title('r en polarisation TE') #permet de donner un titre
plt.legend(loc='best') #permet de placer la légende au meilleur endroit
plt.grid()

### t polarisation TE 

t12 = (2*w1)/(w1 + w2)

plt.figure(2)
plt.plot(theta_deg, t12, label='t', color='red')
plt.xlim((0, 90))
plt.ylim((0, 0.8))
plt.xlabel('Angle d incidence')
plt.title('t en polarisation TE')
plt.legend(loc='best')
plt.grid()

### R polarisation TE 

R = (np.abs(r12))**2 #permet de calculer le coefficients de rÃ©flexion
                     # en Ã©nergie pour 1 dioptre pour la polarisation TE 

plt.figure(3)
plt.plot(theta_deg, R, label='R', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.title('R en polarisation TE')
plt.legend(loc='best')
plt.grid()

### T polarisation TE 

T = (np.abs(t12))**2*((w2 + w2.conjugate())/(w1 + w1.conjugate()))

plt.figure(4)
plt.plot(theta_deg, T, label='T', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.title('T en polarisation TE')
plt.legend(loc='best')
plt.grid()

#%% TE Cas 2 : Calcul et tracé des coefficients r,t,R,T pour TE pour 1 dioptre

### Cas 2
## on refait la même chose que pour le cas 1 mais o modifie n1 et n2
   
n1 = 1.5  # Changement de n1 à 1.5
n2 = 1.0  # Changement de n2 à 1.0
ninc = n1

w1 = wa(ninc, n1, theta)
w2 = wa(ninc, n2, theta)

### r polarisation TE

r12 = (w1 - w2)/(w1 + w2)

plt.figure(5)
plt.plot(theta_deg, r12, label='r', color='red')  
plt.xlim((0, 90))
plt.ylim((-1, 1))
plt.xlabel('Angle d incidence')
plt.title('r en polarisation TE')
plt.legend(loc='best')
plt.grid()

### t polarisation TE 

t12 = (2*w1)/(w1 + w2)

plt.figure(6)
plt.plot(theta_deg, t12, label='t', color='red')
plt.xlim((0, 90))
plt.ylim((0, 2))
plt.xlabel('Angle d incidence')
plt.title('t en polarisation TE')
plt.legend(loc='best')
plt.grid()

### R polarisation TE 

R = (np.abs(r12))**2

plt.figure(7)
plt.plot(theta_deg, R, label='R', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.title('R en polarisation TE')
plt.legend(loc='best')
plt.grid()

### T polarisation TE 

T = (np.abs(t12))**2*((w2 + w2.conjugate())/(w1 + w1.conjugate()))

plt.figure(8)
plt.plot(theta_deg, T, label='T', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.title('T en polarisation TE')
plt.legend(loc='best')
plt.grid()

#%% Vérification de sin theta = n2 / n1

# Chercher le maximum de R
max_index_R = R.tolist().index(max(R))
print("L'abscisse correspondant au maximum de R est :", theta_deg[max_index_R])

# Chercher le minimum de T
min_index_T = T.tolist().index(min(T))
print("L'abscisse correspondant au minimum de T est :", theta_deg[min_index_T])

#%% TM Cas 1 : Calcul et tracé des coefficients r,t,R,T pour TM pour 1 dioptre

#Cas 1
n1 = 1
n2 = 1.5
ninc = n1

w1 = wa(ninc, n1, theta)
w2 = wa(ninc, n2, theta)

E_1 = n1**2
E_2 = n2**2 # ϵ_a est la permittivité relative du milieu a

### r en position TM

r12 = (E_2*w1 - E_1*w2)/(E_2*w1 + E_1*w2) # calcul du coefficient de réflexion
                                          #rab = (ϵb*wa − ϵa*wb)/(ϵb*wa + ϵa*wb)
                                          # en remplaçant a et b par 1 et 2 ici

plt.figure(9)
plt.plot(theta_deg, r12, label='r', color='red')  
plt.xlim((0, 90))
plt.ylim((-1, 0.2))
plt.xlabel('Angle d incidence')
plt.title('r en polarisation TM')
plt.legend(loc='best')
plt.grid()

### t en position TM

t12 = (np.sqrt(E_1/E_2))*((2*E_2*w1)/(E_2*w1 + E_1*w2)) # calcul du coefficient de transmission
                                                        # on utilise la formule de tab
                                                        # en remplaçant a et b par 1 et 2 ici

plt.figure(10)
plt.plot(theta_deg, t12, label='t', color='red')
plt.xlim((0, 90))
plt.ylim((0, 0.8))
plt.xlabel('Angle d incidence')
plt.title('t en polarisation TM')
plt.legend(loc='best')
plt.grid()

### R polarisation TM

R = (np.abs(r12))**2

plt.figure(11)
plt.plot(theta_deg, R, label='R', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.title('R en polarisation TM')
plt.legend(loc='best')
plt.grid()

### T polarisation TM

T = (np.abs(t12))**2*((w2 + w2.conjugate())/(w1 + w1.conjugate()))

plt.figure(12)
plt.plot(theta_deg, T, label='T', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.title('T en polarisation TM')
plt.legend(loc='best')
plt.grid()

#%% Vérification de tan theta = n2 / n1

# Chercher lorsque r = 0 avec la méthode de Newton-Raphson

def fun1(theta):
    n1 = 1.0
    n2 = 1.5
    ninc = n1
    w1 = wa(ninc, n1, theta)
    w2 = wa(ninc, n2, theta)
    E_1 = n1**2
    E_2 = n2**2
    r12 = (E_2 * w1 - E_1 * w2) / (E_2 * w1 + E_1 * w2)
    return r12

def fun1p(theta):
    h = 1e-5  # Petit déplacement pour calculer la dérivée 
    return (fun1(theta + h) - fun1(theta)) / h

erreur = 1e-5
theta_0 = 1.  
theta_old = 0.

while abs(theta_0 - theta_old) > erreur:
    theta_old = theta_0
    theta_0 = theta_old - fun1(theta_old) / fun1p(theta_old)

# Convertir theta_0 en degrés
theta_0_deg = theta_0 * 180 / np.pi

print('Solution théta 0 =', theta_0_deg) 

# Chercher lorsque R = 0 avec la méthode de Newton-Raphson

def fun11(theta):
    return np.abs(r12)**2
    
def fun11p(theta):
    h = 1e-5  # Petit déplacement pour calculer la dérivée 
    return (fun11(theta + h) - fun11(theta)) / h

erreur = 1e-5
theta_0 = 1.  
theta_old = 0.

while abs(theta_0 - theta_old) > erreur:
    theta_old = theta_0
    theta_0 = theta_old - fun1(theta_old) / fun1p(theta_old)

# Convertir theta_0 en degrés
theta_0_deg = theta_0 * 180 / np.pi

print('Solution R théta 0 =', theta_0_deg) 

#%% TM Cas 2 : Calcul et tracé des coefficients r,t,R,T pour TM pour 1 dioptre

#Cas 2
n1 = 1.5
n2 = 1
ninc = n1

w1 = wa(ninc, n1, theta)
w2 = wa(ninc, n2, theta)

E_1 = n1**2
E_2 = n2**2

### r en position TM

r12 = (E_2*w1 - E_1*w2)/(E_2*w1 + E_1*w2)

plt.figure(9)
plt.plot(theta_deg, r12, label='r', color='red')  
plt.xlim((0, 90))
plt.ylim((-1, 1.1))
plt.xlabel('Angle d incidence')
plt.title('r en polarisation TM')
plt.legend(loc='best')
plt.grid()

### t en position TM

t12 = (np.sqrt(E_1/E_2))*((2*E_2*w1)/(E_2*w1 + E_1*w2))

plt.figure(10)
plt.plot(theta_deg, t12, label='t', color='red')
plt.xlim((0, 90))
plt.ylim((0, 3))
plt.xlabel('Angle d incidence')
plt.title('t en polarisation TM')
plt.legend(loc='best')
plt.grid()

### R polarisation TM

R_TM = (np.abs(r12))**2

plt.figure(11)
plt.plot(theta_deg, R_TM, label='R', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1.1))
plt.xlabel('Angle d incidence')
plt.title('R en polarisation TM')
plt.legend(loc='best')
plt.grid()

### T polarisation TM

T_TM = (np.abs(t12))**2*((w2 + w2.conjugate())/(w1 + w1.conjugate()))

plt.figure(12)
plt.plot(theta_deg, T_TM, label='T', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1.1))
plt.xlabel('Angle d incidence')
plt.title('T en polarisation TM')
plt.legend(loc='best')
plt.grid()

#%% Vérification angle theta pour r = 0 et R = 0 pour le cas 2

# Chercher lorsque r = 0 
## on constate qu'il y a deux z�ros... la m�thode de newton-raphson 
#ne fonctionne donc pas tr�s bien

from scipy.optimize import fsolve

# fonction fun2
def fun2(theta):
    n1 = 1.5
    n2 = 1.0
    ninc = n1
    w1 = wa(ninc, n1, theta)
    w2 = wa(ninc, n2, theta)
    E_1 = n1**2
    E_2 = n2**2
    r12 = (E_2 * w1 - E_1 * w2) / (E_2 * w1 + E_1 * w2)
    return r12

# Utilisez fsolve pour trouver le premier z�ro en partant d'une valeur initiale proche de la solution
theta_initial_guess = 0.55  # Remplacez par la valeur initiale souhait�e
theta_solution = fsolve(fun2, theta_initial_guess)

# Convertir theta_solution en degr�s
theta_solution_deg = theta_solution * 180 / np.pi

print('La valeur de theta_0 est :', theta_solution_deg)

### pour R

def R_TM(theta):
    n1 = 1.5
    n2 = 1.0
    ninc = n1
    w1 = wa(ninc, n1, theta)
    w2 = wa(ninc, n2, theta)
    E_1 = n1**2
    E_2 = n2**2
    r12 = (E_2 * w1 - E_1 * w2) / (E_2 * w1 + E_1 * w2)
    R_TM = (np.abs(r12))**2
    return R_TM

# Utilisez fsolve pour trouver le premier z�ro en partant d'une valeur initiale proche de la solution
theta_initial_guess = 0.58  # Remplacez par la valeur initiale souhait�e
theta_solution = fsolve(R_TM, theta_initial_guess)

# Convertir theta_solution en degr�s
theta_solution_deg = theta_solution * 180 / np.pi

print('La valeur de theta_0 pour R_TM = 0 est :', theta_solution_deg)
#%% Verification de sin theta = n2 / n1

# Trouver le maximum de R_TM et son angle associ�
max_R = max(R_TM)
angle_max_R = theta_deg[R_TM.tolist().index(max_R)]
print("L'abscisse correspondant au maximum de R_TM est :", angle_max_R) 

# Trouver le minimum de T_TM et son angle associ�
min_T = min(T_TM)
angle_min_T = theta_deg[T_TM.tolist().index(min_T)]
print("L'abscisse correspondant au minimum de T_TM est :", angle_min_T)


#%% Cas pour 2 dioptres en polarisation TE

def r(r12, r23, w2, d, lmda):
    numerateur1 = (r12 + r23 * np.exp(1j * (2 * np.pi / lmda) * 2 * d * w2))
    denominateur1 = (1 + r12*r23*np.exp(1j * (2 * np.pi / lmda)* 2 * d * w2))
    return numerateur1 / denominateur1

def t(t12, t23, d, lmda, w2, w3, r12, r23):
    numerateur = t12 * t23 * np.exp(1j * (2 * np.pi / lmda) * d * (w2 - w3))
    denominateur = 1 + r12 * r23 * np.exp(1j * (2 * np.pi / lmda) * 2 * d * w2)
    return numerateur / denominateur

n1 = 1.0
n2 = 1.5
n3 = 1.0
ninc = n1

w1 = wa(ninc, n1, theta)
w2 = wa(ninc, n2, theta)
w3 = wa(ninc, n3, theta)

d = 1000
lmda = 500

# ℜ(r) polarisation TE 

r12 = (w1 - w2)/(w1 + w2)
r23 = (w2 - w3)/(w2 + w3)

plt.figure(1)
plt.plot(theta_deg, r(r12, r23, w2, d, lmda), label='r', color='red')
plt.xlim((0, 90))
plt.ylim((-1, 0))
plt.xlabel('Angle d incidence')
plt.title('r en polarisation TE avec 2 dioptres')
plt.legend(loc='best')
plt.grid()

# ℜ(t) en polarisation TE

t12 = (2*w1)/(w1 + w2)
t23 = (2*w2)/(w2 + w3)

plt.figure(2)
plt.plot(theta_deg, t(t12, t23, d, lmda, w2, w3, r12, r23), label='t', color='red')
plt.xlim((0, 90))
plt.ylim((-1, 1))
plt.xlabel('Angle d incidence')
plt.title('t en polarisation TE avec 2 dioptres')
plt.legend(loc='best')
plt.grid()

# R en polarisation TE avec 2 dioptres

R = (np.abs(r(r12, r23, w2, d, lmda)))**2

plt.figure(3)
plt.plot(theta_deg, R, label='R', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.title('R en polarisation TE avec 2 dioptres')
plt.legend(loc='best')
plt.grid()

# T en polarisation TE avec 2 dioptres

T = (np.abs(t(t12, t23, d, lmda, w2, w3, r12, r23)))**2*((w3 + w3.conjugate())/(w1 + w1.conjugate()))

plt.figure(4)
plt.plot(theta_deg, T, label='T', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.title('T en polarisation TE avec 2 dioptres')
plt.legend(loc='best')
plt.grid()

#%% Phénomène de frustration de la réflexion totale interne TE
# avec deux dioptres

n1 = 1.5
n2 = 1.0
n3 = 1.5
ninc = n1

w1 = wa(ninc, n1, theta)
w2 = wa(ninc, n2, theta)
w3 = wa(ninc, n3, theta)

d1 = 10
d2 = 100
d3 = 1000
lmda = 720

T1 = (np.abs(t(t12, t23, d1, lmda, w2, w3, r12, r23)))**2*((w3 + w3.conjugate())/(w1 + w1.conjugate()))
T2 = (np.abs(t(t12, t23, d2, lmda, w2, w3, r12, r23)))**2*((w3 + w3.conjugate())/(w1 + w1.conjugate()))
T3 = (np.abs(t(t12, t23, d3, lmda, w2, w3, r12, r23)))**2*((w3 + w3.conjugate())/(w1 + w1.conjugate()))

plt.figure(5)
plt.plot(theta_deg, T1, label='e = 10', color='blue')
plt.plot(theta_deg, T2, label='e = 100', color='red')
plt.plot(theta_deg, T3, label='e = 1000', color='green')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.ylabel('Intensité transmise')
plt.title('T en polarisation TE avec 2 dioptres')
plt.legend(loc='best')
plt.grid()

#%%Cas particulier : excitation d'un plasmon de surface 

n1 = 1.5
n2 = 0.24*(1 + 12.875*1j)
n3 = 1
ninc = n1

d = 40
lmda = 500

w1 = wa(ninc, n1, theta)
w2 = wa(ninc, n2, theta)
w3 = wa(ninc, n3, theta)

ϵ_1 = n1**2
ϵ_2 = n2**2
ϵ_3 = n3**2

### r en position TM

r12 = (ϵ_2*w1 - ϵ_1*w2)/(ϵ_2*w1 + ϵ_1*w2)
r23 = (ϵ_3*w2 - ϵ_2*w3)/(ϵ_3*w2 + ϵ_2*w3)

plt.figure(6)
plt.plot(theta_deg, np.abs(r(r12, r23, w2, d, lmda)), label='r', color='red')  
plt.xlim((0, 90))
plt.ylim((0.1, 1))
plt.xlabel('Angle d incidence')
plt.ylabel('Abs(r)')
plt.title('|r| en polarisation TM')
plt.legend(loc='best')
plt.grid()

### t en position TM

t12 = (np.sqrt(ϵ_1/ϵ_2))*((2*ϵ_2*w1)/(ϵ_2*w1 + ϵ_1*w2))
t23 = (np.sqrt(ϵ_2/ϵ_3))*((2*ϵ_3*w2)/(ϵ_3*w2 + ϵ_2*w3))

plt.figure(7)
plt.plot(theta_deg, np.abs(t(t12, t23, d, lmda, w2, w3, r12, r23)), label='|t|', color='red')
plt.xlim((0, 90))
plt.ylim((0, 7))
plt.xlabel('Angle d incidence')
plt.ylabel('Abs(t)')
plt.title('t en polarisation TM')
plt.legend(loc='best')
plt.grid()

### R et T en position TM

R = (np.abs(r(r12, r23, w2, d, lmda)))**2
T = (np.abs(t(t12, t23, d, lmda, w2, w3, r12, r23)))**2*((w3 + w3.conjugate())/(w1 + w1.conjugate()))

plt.figure(8)
plt.plot(theta_deg, R, label='R', color='red')
plt.plot(theta_deg, T, label='T', color='green')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.ylabel('R,T')
plt.title('R et T en polarisation TM')
plt.legend(loc='best')
plt.grid()

### R+T en position TM

u = R + T

plt.figure(9)
plt.plot(theta_deg, u, label='R', color='red')
plt.xlim((0, 90))
plt.ylim((0, 1))
plt.xlabel('Angle d incidence')
plt.ylabel('R + T')
plt.title('R+T en polarisation TM')
plt.legend(loc='best')
plt.grid()

#%%Cas particulier : variation de e et graphique 3D




