import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
# https://colour.readthedocs.io/en/develop/tutorial.html
import colour
from colour.plotting import *
from colour.models import XYZ_to_sRGB



# Plotting the *CIE 1931 Chromaticity Diagram*.
# The argument *standalone=False* is passed so that the plot doesn't get
# displayed and can be used as a basis for other plots.


## fonctionbs trichromatiques
lc, xc, yc, zc = np.loadtxt('ciexyz31_1.txt',unpack = True, delimiter = ',')

# on va prendre lc comme référence pour les longueurs d'onde, en nm

vert_yc = np.vstack(yc)
lE, E = np.loadtxt('D65by1.dat', unpack = True, delimiter = ',')

#lAg, nAg, kappaAg = np.loadtxt('nk_cu_palik.txt', unpack = True, delimiter = ' ')
lAg, nAg, kappaAg = np.loadtxt('ag_jc.dat', unpack = True, delimiter = '\t')

R = lambda n, kappa : np.abs((1 - (n + kappa*1j))/(1 + (n + kappa*1j)))**2

R_inter = CubicSpline(lAg*1e3, R(nAg, kappaAg))


plt.plot(lAg*1e3, R(nAg, kappaAg), 'o')
plt.plot(lAg*1e3, R_inter(lAg*1e3))

lambda_f = np.arange(360, 831, 1)
print(len(lambda_f))
print(len(vert_yc), len(E[60:]))

k = 100/(E[60:]@vert_yc)
X, Y, Z = 0, 0, 0
for i in range(len(lambda_f)):
    X += k*(R_inter(lambda_f[i])*E[60:][i]*xc[i])
    Y += k*(R_inter(lambda_f[i])*E[60:][i]*yc[i])
    Z += k*(R_inter(lambda_f[i])*E[60:][i]*zc[i])

print(X, Y, Z)
x = X/(X+Y+Z)
y = Y/(X+Y+Z)
z = Z/(X+Y+Z)

print(x, y)

## illuminant D65
plot_chromaticity_diagram_CIE1931(standalone=False)
x_w = 0.31259787
y_w = 0.32870029

XYZ = np.array([X[0], Y[0], Z[0]])

print(XYZ)

RGB = XYZ_to_sRGB(XYZ / 100)
RGB_255 = RGB * 255

print("Valeurs RGB :", RGB, "\n RDB 255 :", RGB_255)


#plt.plot(x,y,'o', color='black')
plt.scatter(np.array(x), np.array(y), color=RGB, s=100, edgecolors='black', linewidths=1)



plt.show()
"""
Attention, toutes les données ne sont pas dans les mêmes unités
pour l'argent, lambda en micrometres Re(n) Im(n)
pour l'or, lambda en nm Re(epsilon) Im(epsilon)
pour le cuivre, lambda en micrometres Re(n) Im(n)

De plus ce ne sont pas forcément les mêmes longueurs d onde que les fonctions trichromatiques
Donc il va falloir ruser !

"""
