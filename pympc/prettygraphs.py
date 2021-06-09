from numpy import linalg
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.external.husl import f
import mpc

sns.set_style("whitegrid")
sns.set_context("paper")
cmap = sns.color_palette()
fig, ax = plt.subplots()

A = np.random.randn(2,2) 
A = .5 * (A + A.T) + np.eye(2) * 2
b = np.random.rand(2) * 4
D = np.eye(2)
d = np.abs(np.random.rand(2)) * -2

constraing_lim = np.linalg.solve(D,d)

def J(x,y):
    v = np.array([[y],[x]])
    return -v.T @ b + .5 * v.T @ A @ v

ylim = 10
xlim = 10
dimcount = 100
x, y = np.linspace(-xlim,xlim,dimcount),np.linspace(-ylim,ylim,dimcount)
xx, yy = np.meshgrid(x, y)
z = np.zeros((dimcount,dimcount))
for xc, xval in enumerate(x):
    for yc, yval in enumerate(y):
        z[xc,yc] = J(xval, yval)
plt.contourf(xx,yy,z)


l1 = plt.hlines(constraing_lim[1], -10, constraing_lim[0], color="white", label="Constraints")
l2 = plt.vlines(constraing_lim[0], -10, constraing_lim[1], color="white")
soltn = mpc.solve_qp_ip_ext(A, b, D, d,full=True)
print(soltn)
plt.scatter(soltn[0][0], soltn[0][1], label="Solution", color="white")
iters = mpc.solve_qp_ip(A, b, D, d, iters=True)
print(iters)
plt.plot(iters[:,0],iters[:,1],color=cmap[1],label="Solver Path")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Typical inequality-constrained QP problem")
cbar = plt.colorbar()
cbar.set_label("Cost Function J")
plt.legend()
plt.show()