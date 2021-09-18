import numpy as np 
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from mpl_toolkits import mplot3d

l = 10
m = 10
n = 10
N = l*m*n

x_0 = 0
x_1l = 1
y_0 = 0
y_1m = 1
z_0 = 0
z_1n = 1

x_i = np.linspace(x_0, x_1l, l, endpoint="True")
y_j = np.linspace(y_0, y_1m, m, endpoint="True")
z_k = np.linspace(z_0, z_1n, n, endpoint="True")

dx = (x_1l - x_0) / l
dy = (y_1m - y_0) / m
dz = (z_1n - z_0) / n

# X difference diags
Dx_lead = np.full(N, -2/(dx**2))
Dx_off = np.full(N-1, 1/(dx**2))
for a in range(1, m*n):
    Dx_off[a*l - 1] = 0

Dx = scipy.sparse.diags([
    Dx_off,
    Dx_lead,
    Dx_off,
    ],
    [-1, 0, 1])
    
# Y difference Diags
Dy_lead = np.full(N, -2/(dy**2))
Dy_off = np.full(N-l, 1/(dy**2))
for a in range(1,):
    Dy_off[a*m - 1] = 0

Dy = scipy.sparse.diags([
    Dy_off,
    Dy_lead,
    Dy_off,
    ],
    [-l, 0, l])

# Z difference diags
Dz_lead = np.full(N, -2/(dz**2))
Dz_off = np.full(N -(l*m), 1/(dz**2))

Dz = scipy.sparse.diags([
    Dz_off,
    Dz_lead,
    Dz_off,
    ],
    [-(l*m), 0, l*m])
#Potential oeprator



#Adding the operators together to get the hailtonian 
hamiltonian = -(Dx + Dy + Dz)

# Finding the eigenvalues and vectoer
evals, evec = scipy.sparse.linalg.eigsh(hamiltonian, k=3, which="SM")

print(evals)

# Plotting the wave function suraface
psi = evec[:, 0]    # chooosing which wave function to plot

print(len(psi) == N) 
y = 0.9 * np.max(np.abs(psi))
psi = np.reshape(psi, (l, m, n))

print(y)

x, y, z, phi = np.array([]), np.array([]), np.array([]), np.array([])

for index, p in np.ndenumerate(psi):
    if np.abs(p) <= y:
        np.append(phi, p)
        np.append(x, x_i[index[0]])
        np.append(y, y_j[index[1]])
        np.append(z, z_k[index[2]])
    else:
        pass

print(x, y, z, phi)

x, y, z = np.meshgrid(x_i, y_j, z_k)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
wave_functions = ax.scatter(x, y, z, c=psi.flat)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

cbar = plt.colorbar(wave_functions)

plt.show()