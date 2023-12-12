import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
# https://colour.readthedocs.io/en/develop/tutorial.html
import colour
from colour.plotting import *



# Plotting the *CIE 1931 Chromaticity Diagram*.
# The argument *standalone=False* is passed so that the plot doesn't get
# displayed and can be used as a basis for other plots.


## fonctionbs trichromatiques
lc, xc, yc, zc = np.loadtxt('ciexyz31_1.txt',unpack = True, delimiter = ',')

# on va prendre lc comme référence pour les longueurs d'onde, en nm

vert_yc = np.vstack(yc)
lE, E = np.loadtxt('D65by1.dat', unpack = True, delimiter = ',')

lOr, epsilonOr, kappaOr = np.loadtxt('gold_palik_eps.txt', unpack = True)

n = np.sqrt(epsilonOr + kappaOr*1j)

R = lambda n : np.abs((1 - n)/(1 + n))**2

R_inter = CubicSpline(lOr, R(n))


plt.plot(lOr, R(n), 'o')
plt.plot(lOr, R_inter(lOr))

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

print(x, y)

## illuminant D65
plot_chromaticity_diagram_CIE1931(standalone=False)
x_w = 0.31259787
y_w = 0.32870029
plt.plot(x,y,'o', color='black')


plt.show()
"""
Attention, toutes les données ne sont pas dans les mêmes unités
pour l'argent, lambda en micrometres Re(n) Im(n)
pour l'or, lambda en nm Re(epsilon) Im(epsilon)
pour le cuivre, lambda en micrometres Re(n) Im(n)

De plus ce ne sont pas forcément les mêmes longueurs d onde que les fonctions trichromatiques
Donc il va falloir ruser !

"""
