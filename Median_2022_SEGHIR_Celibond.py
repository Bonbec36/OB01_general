import numpy as np
import matplotlib.pyplot as plt

def question_1():
    V = 500
    r = np.linspace(2, 10, 200)
    S = lambda r, V: 2*(np.pi*r**2 + V/r) 
    h = lambda r : V/(np.pi*r**2)

    r0 = r[S(r, V).tolist().index(min(S(r, V)))]
    h0 = h(r0)

    print(r0, h0)

    plt.xlabel('Rayon en cm') #permet de nommer les axes
    plt.ylabel('Surface en cm^2') #permet de nommer les axes
    plt.title('Surface en fonction du Rayon pour V = 500mL') #permet de donner un titre
    plt.plot(r, S(r, V), 'r')
    plt.show()

def question_2():
    M0 = 6.24
    M1 = 0.0172
    C1 = lambda e : (2*e - 0.25*e**3)
    C2 = lambda e : 1.25*e**2
    L0 = 4.89
    R2 = lambda epsilon : -(np.tan(epsilon/2)**2)
    R4 = lambda epsilon : 0.5*np.tan(epsilon/2)**4

    M = lambda d : M0 + M1 * d
    C = lambda m : C1*np.sin(m) + C2*np.sin(2*m)
    L = lambda c, d : L0 + c + M1 *d
    R = lambda l : R2*np.sin(2*l) + R4*np.sin(4*l)

    dt = lambda k, c, r : k*(c+r)

    K = dt/(C+R)
    



question_2()