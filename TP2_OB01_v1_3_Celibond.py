import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from numpy.linalg import inv, matrix_power



def chapitre_3_1_1():
    n1 = 1.5
    n2 = 1
    
    theta = np.linspace(0, np.pi/2, 200)
    
    wa = lambda na, ni, theta: np.sqrt(na**2-(ni*np.sin(theta))**2+0j)
    r_ab = lambda w_a, w_b: (w_a-w_b)/(w_a+w_b)
    t_ab = lambda w_a, w_b : 2*w_a/(w_a+w_b)
    
    rm_ab = lambda na, nb, w_a, w_b : ((nb**2)*w_a-(na**2)*w_b)/((nb**2)*w_a+(na**2)*w_b)
    tm_ab = lambda na, nb, w_a, w_b : np.sqrt((na**2)/(nb**2))*(2*(nb**2)*w_a)/((nb**2)*w_a+(na**2)*w_b)
    
    rad_to_deg = lambda rad: int(rad*180/np.pi)
    
    r = r_ab(wa(n1, n1, theta), wa(n2, n1, theta))
    t = t_ab(wa(n1, n1, theta), wa(n2, n1, theta))
    
    rm = rm_ab(n1, n2, wa(n1, n1, theta), wa(n2, n1, theta))
    tm = tm_ab(n1, n2, wa(n1, n1, theta), wa(n2, n1, theta))
    
    
    R = np.square(np.abs(r))
    T = np.square(np.abs(t))*(wa(n2, n1, theta)+np.conj(wa(n2, n1, theta)))/(wa(n1, n1, theta)+np.conj(wa(n1, n1, theta)))
    
    Rm = np.square(np.abs(rm))
    Tm = np.square(np.abs(tm))*(wa(n2, n1, theta)+np.conj(wa(n2, n1, theta)))/(wa(n1, n1, theta)+np.conj(wa(n1, n1, theta)))
    
    plt.plot(theta, Rm)
    #print(theta[R.tolist().index(max(R))])
    #print(T.tolist().index(min(T)))
    #print(theta[Rm.tolist().index(min(Rm))])
    plt.xlim([0, np.pi/2])
    degree_lab = [rad_to_deg(i) for i in plt.xticks()[0]]
    plt.xticks(plt.xticks()[0], labels=(degree_lab))
    plt.show()
    
def chapitre_3_2_1():
    n1 = 1
    n2 = 1.5
    n3 = 1
    lmd = 500
    d = 1000
    
    theta = np.linspace(0, np.pi/2, 200)
    
    wa = lambda na, ni, theta: np.sqrt(na**2-(ni*np.sin(theta))**2+0j)
    r_ab = lambda w_a, w_b: (w_a-w_b)/(w_a+w_b)
    t_ab = lambda w_a, w_b : 2*w_a/(w_a+w_b)
    
    f_r = lambda r1, r2, w: (r1+r2*np.exp(1j*2*(np.pi/lmd)*2*d*w))/(1+r1*r2*np.exp(1j*2*(np.pi/lmd)*2*d*w))
    f_t = lambda t1, t2, r1, r2, w1, w2: (t1*t2*np.exp(1j*2*(np.pi/lmd)*d*(w1-w2)))/(1+r1*r2*np.exp(1j*2*(np.pi/lmd)*2*d*w1))
    
    rad_to_deg = lambda rad: int(rad*180/np.pi)
    
    r_12 = r_ab(wa(n1, n1, theta), wa(n2, n1, theta))
    r_23 = r_ab(wa(n2, n1, theta), wa(n3, n1, theta))
    
    
    r = f_r(r_ab(wa(n1, n1, theta), wa(n2, n1, theta)), r_ab(wa(n2, n1, theta), wa(n3, n1, theta)), wa(n2, n1, theta))
    t = f_t(t_ab(wa(n1, n1, theta), wa(n2, n1, theta)), t_ab(wa(n2, n1, theta), wa(n3, n1, theta)), r_ab(wa(n1, n1, theta), wa(n2, n1, theta)),
            r_ab(wa(n2, n1, theta), wa(n3, n1, theta)),  wa(n2, n1, theta),  wa(n3, n1, theta))

    R = np.square(np.abs(r))
    T = np.square(np.abs(t))*(wa(n3, n1, theta)+np.conj(wa(n3, n1, theta)))/(wa(n1, n1, theta)+np.conj(wa(n1, n1, theta)))
    
    plt.plot(theta, T)
    plt.xlim([0, np.pi/2])
    degree_lab = [rad_to_deg(i) for i in plt.xticks()[0]]
    plt.xticks(plt.xticks()[0], labels=(degree_lab))
    plt.show()
    
def chapitre_3_2_2():
    n1 = 1.5
    n2 = 1
    n3 = 1.5
    lmd = 720
    d = 10
    
    theta = np.linspace(0, np.pi/2, 200)
    
    wa = lambda na, ni, theta: np.sqrt(na**2-(ni*np.sin(theta))**2+0j)
    r_ab = lambda w_a, w_b: (w_a-w_b)/(w_a+w_b)
    t_ab = lambda w_a, w_b : 2*w_a/(w_a+w_b)
    
    f_r = lambda r1, r2, w, d: (r1+r2*np.exp(1j*2*(np.pi/lmd)*2*d*w))/(1+r1*r2*np.exp(1j*2*(np.pi/lmd)*2*d*w))
    f_t = lambda t1, t2, r1, r2, w1, w2, d: (t1*t2*np.exp(1j*2*(np.pi/lmd)*d*(w1-w2)))/(1+r1*r2*np.exp(1j*2*(np.pi/lmd)*2*d*w1))
    
    rad_to_deg = lambda rad: int(rad*180/np.pi)
    
    r_12 = r_ab(wa(n1, n1, theta), wa(n2, n1, theta))
    r_23 = r_ab(wa(n2, n1, theta), wa(n3, n1, theta))
    
    
    r_10 = f_r(r_ab(wa(n1, n1, theta), wa(n2, n1, theta)), r_ab(wa(n2, n1, theta), wa(n3, n1, theta)), wa(n2, n1, theta), 10)
    t_10 = f_t(t_ab(wa(n1, n1, theta), wa(n2, n1, theta)), t_ab(wa(n2, n1, theta), wa(n3, n1, theta)), r_ab(wa(n1, n1, theta), wa(n2, n1, theta)),
            r_ab(wa(n2, n1, theta), wa(n3, n1, theta)),  wa(n2, n1, theta),  wa(n3, n1, theta), 10)
    
    r_100 = f_r(r_ab(wa(n1, n1, theta), wa(n2, n1, theta)), r_ab(wa(n2, n1, theta), wa(n3, n1, theta)), wa(n2, n1, theta), 100)
    t_100 = f_t(t_ab(wa(n1, n1, theta), wa(n2, n1, theta)), t_ab(wa(n2, n1, theta), wa(n3, n1, theta)), r_ab(wa(n1, n1, theta), wa(n2, n1, theta)),
            r_ab(wa(n2, n1, theta), wa(n3, n1, theta)),  wa(n2, n1, theta),  wa(n3, n1, theta), 100)
    
    r_1000 = f_r(r_ab(wa(n1, n1, theta), wa(n2, n1, theta)), r_ab(wa(n2, n1, theta), wa(n3, n1, theta)), wa(n2, n1, theta), 1000)
    t_1000 = f_t(t_ab(wa(n1, n1, theta), wa(n2, n1, theta)), t_ab(wa(n2, n1, theta), wa(n3, n1, theta)), r_ab(wa(n1, n1, theta), wa(n2, n1, theta)),
            r_ab(wa(n2, n1, theta), wa(n3, n1, theta)),  wa(n2, n1, theta),  wa(n3, n1, theta), 1000)
    

    R = lambda r : np.square(np.abs(r))
    T = lambda t : np.square(np.abs(t))*(wa(n3, n1, theta)+np.conj(wa(n3, n1, theta)))/(wa(n1, n1, theta)+np.conj(wa(n1, n1, theta)))
    
    plt.plot(theta, T(t_10), 'b')
    plt.plot(theta, T(t_100), 'r', linestyle='--')
    plt.plot(theta, T(t_1000), 'g', linestyle='--')
    
    plt.xlim([0, np.pi/2])
    degree_lab = [rad_to_deg(i) for i in plt.xticks()[0]]
    plt.xticks(plt.xticks()[0], labels=(degree_lab))
    plt.show()
    
def chapitre_3_3():
    n1 = 1.5
    n2 = 0.24*(1 + 12.875j)
    n3 = 1
    lmd = 500
    d = 40
    
    theta = np.linspace(0, np.pi/2, 200)
    
    wa = lambda na, ni, theta: np.sqrt(na**2-(ni*np.sin(theta))**2)
    r_ab = lambda w_a, w_b, na, nb: (w_a*nb**2-w_b*na**2)/(w_a*nb**2+w_b*na**2)
    t_ab = lambda w_a, w_b, na, nb : np.sqrt(na**2/nb**2)*(2*nb**2*w_a/(w_a*nb**2+w_b*na**2))
    
    f_r = lambda r1, r2, w: (r1+r2*np.exp(1j*2*(np.pi/lmd)*2*d*w))/(1+r1*r2*np.exp(1j*2*(np.pi/lmd)*2*d*w))
    f_t = lambda t1, t2, r1, r2, w1, w2: (t1*t2*np.exp(1j*2*(np.pi/lmd)*d*(w1-w2)))/(1+r1*r2*np.exp(1j*2*(np.pi/lmd)*2*d*w1))
    
    rad_to_deg = lambda rad: int(rad*180/np.pi)
    
    #r_12 = r_ab(wa(n1, n1, theta), wa(n2, n1, theta))
    #r_23 = r_ab(wa(n2, n1, theta), wa(n3, n1, theta))
    
    
    r = f_r(r_ab(wa(n1, n1, theta), wa(n2, n1, theta), n1, n2), r_ab(wa(n2, n1, theta), wa(n3, n1, theta), n2, n3), wa(n2, n1, theta))
    t = f_t(t_ab(wa(n1, n1, theta), wa(n2, n1, theta), n1, n2), t_ab(wa(n2, n1, theta), wa(n3, n1, theta), n2, n3), r_ab(wa(n1, n1, theta), wa(n2, n1, theta), n1, n2),
    r_ab(wa(n2, n1, theta), wa(n3, n1, theta), n2, n3),  wa(n2, n1, theta),  wa(n3, n1, theta))

    R = np.square(np.abs(r))
    T = np.square(np.abs(t))*(wa(n3, n1, theta)+np.conj(wa(n3, n1, theta)))/(wa(n1, n1, theta)+np.conj(wa(n1, n1, theta)))
    
    plt.plot(theta, R+T)
    plt.xlim([0, np.pi/2])
    degree_lab = [rad_to_deg(i) for i in plt.xticks()[0]]
    plt.xticks(plt.xticks()[0], labels=(degree_lab))
    plt.show()
    

def chapitre_3_3_2():
    n1 = 1.5
    n2 = 0.24*(1 + 12.875* 1j)
    n3 = 1
    lmd = 500
    
    d = np.linspace(10, 100, 500)
    theta = np.linspace(0, np.pi/2, 500)
    
    THETA, D = np.meshgrid(theta, d)
    
    wa = lambda na, ni, theta: np.sqrt(na**2-(ni*np.sin(theta))**2+0j)
    r_ab = lambda w_a, w_b, na, nb: (w_a*nb**2-w_b*na**2)/(w_a*nb**2+w_b*na**2)
    t_ab = lambda w_a, w_b, na, nb : np.sqrt(na**2/nb**2)*(2*nb**2*w_a/(w_a*nb**2+w_b*na**2))
    
    f_r = lambda r1, r2, w: (r1+r2*np.exp(1j*2*(np.pi/lmd)*2*D@w))/(1+r1*r2*np.exp(1j*2*(np.pi/lmd)*2*D@w))
    f_t = lambda t1, t2, r1, r2, w1, w2: (t1*t2*np.exp(1j*2*(np.pi/lmd)*D*(w1-w2)))/(1+r1*r2*np.exp(1j*2*(np.pi/lmd)*2*D*w1))
    
    rad_to_deg = lambda rad: int(rad*180/np.pi)
    
    #r_12 = r_ab(wa(n1, n1, theta), wa(n2, n1, theta))
    #r_23 = r_ab(wa(n2, n1, theta), wa(n3, n1, theta))

    
    r = f_r(r_ab(wa(n1, n1, THETA), wa(n2, n1, THETA), n1, n2), r_ab(wa(n2, n1, THETA), wa(n3, n1, THETA), n2, n3), wa(n2, n1, THETA))
    t = f_t(t_ab(wa(n1, n1, THETA), wa(n2, n1, THETA), n1, n2), t_ab(wa(n2, n1, THETA), wa(n3, n1, THETA), n2, n3), r_ab(wa(n1, n1, THETA), wa(n2, n1,THETA), n1, n2),
    r_ab(wa(n2, n1, THETA), wa(n3, n1, THETA), n2, n3),  wa(n2, n1, THETA),  wa(n3, n1, THETA))
    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    
    surf = ax.plot_surface(THETA, D,np.abs(t), cmap='plasma',
                       linewidth=0, antialiased=False, rcount=400, ccount=400)
    ax.contourf(THETA, D, np.abs(t), levels=25, zdir='z', offset=0, cmap='plasma')
    #plt.plot(theta, np.abs(t))
    plt.xlim([0, np.pi/2])
    degree_lab = [rad_to_deg(i) for i in plt.xticks()[0]]
    plt.xticks(plt.xticks()[0], labels=(degree_lab))
    plt.savefig("myImage.png", format="png", dpi=1500)
    plt.show()
    
    
def chapitre_4():
    n1 = 1.5
    n2 = 1
    N = 2
    theta =  np.pi/4
    
    sigmad = np.linspace(0.48, 0.66, 10)
    
    wa = lambda na, ni, theta: np.sqrt(na**2-(ni*np.sin(theta))**2+0j)
    Ma = lambda wa: np.array([[1, 1], [-wa, wa]])
    Pa = lambda wa, sigmad: np.array([[np.exp(1j*2*np.pi*wa*sigmad), 0], [0, np.exp(-1j*2*np.pi*wa*sigmad)]])
    
    
    M1 = Ma(wa(n1, n1, theta))
    M2 = Ma(wa(n2, n1, theta))
    #P1 = inv(Pa(wa(n1, n1, theta), lmd, d))
    P1_list = [Pa(wa(n1, n1, theta), i) for i in sigmad]
    #P2 = inv(Pa(wa(n2, n1, theta), lmd, d))
    P2_list = [Pa(wa(n2, n1, theta), i) for i in sigmad]
                 
    
    
    #Mtot = inv(M1)*M2*inv(P2)*inv(M2)*(M1*inv(P1)*inv(M1)*M2*inv(P2)*inv(M2))**(N-1)*M1
    
    
    Mtot_list = [inv(M1) @ M2 @ inv(P2_list[i]) @ inv(M2) @ matrix_power((M1 @ inv(P1_list[i]) @ inv(M1) @ M2 @ inv(P2_list[i]) @ inv(M2)), (N-1)) @ M1 for i in range(len(sigmad))]
    print(Mtot_list[5])
    
    list_r = [x[1][0]/x[0][0] for x in Mtot_list]

    
    plt.plot(sigmad, np.square(np.abs(list_r)))
    
chapitre_3_3_2()