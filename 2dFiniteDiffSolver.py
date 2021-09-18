import numpy as np 
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

n = 100
m = 100
N = n*m

x_0 = -1
x_1n = 1

y_0 = -1
y_1m = 1

x_i = np.linspace(x_0, x_1n, n, endpoint="True")
y_i = np.linspace(y_0, y_1m, m, endpoint="True")

dx = (x_1n - x_0) / n
dy = (y_1m - y_0) / m

# x difference matrix
Dx_lead = np.full(N, -2/(dx**2))
Dx_off = np.full(N-1, 1/(dx**2))
for a in range(1, m):
    Dx_off[a*n -1] = 0
        
Dx = scipy.sparse.diags([
    Dx_off,
    Dx_lead,
    Dx_off,
    ],
    [-1, 0, 1])

# y difference matrix
Dy_lead = np.full(N, -2/(dy**2))
Dy_off = np.full(N-n, 1/(dy**2))

Dy = scipy.sparse.diags([
    Dy_off,
    Dy_lead,
    Dy_off,
    ],
    (-n, 0, n)
    )

# Potential Matrix
# harmonic potential

V_x = x_i**2
V_y = y_i**2
V_lead = np.reshape(np.outer(V_x, V_y), N)
V = scipy.sparse.diags(V_lead, 0)

# Evaluating the Eigenvectors and values
evals, evec = scipy.sparse.linalg.eigs(-(Dx+Dy) + V,k=10, which="SM")

print(evals)

# Plotting
X, Y = np.meshgrid(x_i, y_i)
for i in range(len(evals)-1, 0, -1):
    plt.pcolormesh(X, Y, np.real(np.reshape(evec[:, i], (m, n))), shading="auto")
    plt.colorbar()
    plt.show()
    
#plt.legend()
#plt.show()
