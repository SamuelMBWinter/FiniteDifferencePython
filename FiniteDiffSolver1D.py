import numpy as np 
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


N = 2000        # 20000 on home cmputer is max, 40000 on library computers ~ 2-3 mins, 80000 >30 mins for harmonic potential

x_0 = -10
x_N = 10

x_i = np.linspace(x_0, x_N, N, endpoint="True")

h = (x_N - x_0) / N

lead_diagonal = np.full(N, 1/(h**2) + 100*(x_i*x_i))
off_diagonal = np.full((N - 1), -1/(2 * h**2))

Dx = scipy.sparse.diags([off_diagonal, lead_diagonal, off_diagonal], [-1, 0, 1])

evals, evec = scipy.sparse.linalg.eigsh(Dx, k=100, which="SM")

for i in range(0, 4):
    plt.plot(x_i, evec[:, i], label=f"$\psi {i + 1}$")
    #plt.plot(x_i, evec[:, i]**2, label=f"$\psi^{2}_{i + 1}$ ")

print(evals[0:100])

plt.legend()
plt.show()


