import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import *


def function_T(T_precedente, delta_t, delta_x):
    Tn_list = []
    for i in range(len(T_precedente)):
        A = T_precedente[i]
        if i == 0:
            B = 0
            C = T_precedente[i + 1]
            Tn_list.append(A + (delta_t*(C + B - 2*A))/delta_x**2)
        elif i == len(T_precedente) - 1:
            C = 0
            B = T_precedente[i - 1]
            Tn_list.append(A + (delta_t*(C + B - 2*A))/delta_x**2)
        else:
            B = T_precedente[i - 1]
            C = T_precedente[i + 1]
            Tn_list.append(A + (delta_t*(C + B - 2*A))/delta_x**2)
    return Tn_list   
        





    
def Ex_1_1():
    T = lambda t, x : np.exp(-t*(np.pi)**2)*np.sin(np.pi*x)
        
    dx = 1e-3
    dt = 5e-5
    n = 50
    
    x = np.arange(0, 1+dx, dx)
    T0_list = [T(0, i) for i in x]
    T_list = [T0_list]
    for i in range(n):
        T_list.append(function_T(T_list[-1], dt, dx))
    
    m = 10
    plt.plot(x, T_list[m], label='Explicite',  color='b')
    plt.plot(x, T(m*dt, x), label='Analytique',  color='r')
    plt.ylim((0, 1.01))
    plt.xlabel('Position')
    plt.ylabel('Température')
    plt.title(f"Température par rapport à x à dx = {dx}")
    plt.legend(loc='best')
    plt.show()

def Ex_1_3():
    T = lambda t, x : np.exp(-t*(np.pi)**2)*np.sin(np.pi*x)  
    n = 100
    etape = 100
    
    nT = n
    nX = n
    
    dx = 1e-2
    dt = 1e-4
    
    r = dt/(dx**2)
    x_int = np.linspace(0, 1, nX)
    
    Tn = np.vstack(T(0, x_int))
    #M = np.zeros((nX, nT))
    
    A = []
    
    for x in range(nX):
        if x == 0:
            ligne = [1 if i == 0 else 0 for i in range(nT)]
        elif x == nX-1:
            ligne = [1 if i == nT-1 else 0 for i in range (nT)]
        else:
            ligne = []
            for t in range(nT):
                if t == x-1 or t == x + 1:
                    ligne.append(-r)
                elif t == x:
                    ligne.append(1 + 2*r)
                else:
                    ligne.append(0)
    
        A.append(ligne)
        
    Tnm1 = Tn
    
    for i in range(etape):
        T1 = inv(A)@Tnm1
        Tnm1 = T1
    
    #print(T(etape*dt, x_int))
    plt.plot(x_int, Tnm1, label='Implicite',  color='b')
    plt.plot(x_int, T(etape*dt, x_int), label='Analytique',  color='r')
    plt.ylim((0, 1.01))
    plt.xlabel('Position')
    plt.ylabel('Température')
    plt.title(f"Température par rapport à x à dt = {etape}dt")
    plt.legend(loc='best')
    plt.show()


def Ex_1_3_bis():
    T = lambda t, x : np.exp(-t*(np.pi)**2)*np.sin(np.pi*x)  
    n = 200
    etape = 0
    
    nT = n
    nX = n
    
    dx = 1e-2
    dt = 1e-1
    
    r = dt/(dx**2)
    x_int = np.linspace(0, 1, nX)
    
    Tn = np.vstack(T(0, x_int))
    #M = np.zeros((nX, nT))
    T_list = [Tn]
    A = []
    
    for x in range(nX):
        if x == 0:
            ligne = [1 if i == 0 else 0 for i in range(nT)]
        elif x == nX-1:
            ligne = [1 if i == nT-1 else 0 for i in range (nT)]
        else:
            ligne = []
            for t in range(nT):
                if t == x-1 or t == x + 1:
                    ligne.append(-r)
                elif t == x:
                    ligne.append(1 + 2*r)
                else:
                    ligne.append(0)
    
        A.append(ligne)
    
    for i in range(etape):
        T1 = inv(A)@T_list[-1]
        T_list.append(T1)
    
    temps = [i * dt * etape for i in range(len(T_list))]

    T_matrix = np.array(T_list)[:, :, 0].T
    X, Y = np.meshgrid(temps, x_int)

    plt.plot(x_int, T_list[etape], label='Implicite',  color='b')
    plt.ylim((0, 1.01))
    plt.xlabel('Position')
    plt.ylabel('Température')
    plt.title(f"Température par rapport à x à dt = {etape}dt")
    plt.legend(loc='best')
    plt.show()

    plt.contourf(X, Y, T_matrix, cmap='viridis', levels=20)
    plt.colorbar(label='Température')
    plt.xlabel('Temps')
    plt.ylabel('Position')
    plt.title('Température en fonction du temps et de la position')
    plt.show()

    """ 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, T_matrix, cmap='viridis', rstride=1, cstride=1, edgecolor='none')
    ax.set_xlabel('Position')
    ax.set_ylabel('Temps')
    ax.set_zlabel('Température')
    ax.set_title('Température en fonction du temps et de la position')
    plt.show()"""

Ex_1_3_bis
