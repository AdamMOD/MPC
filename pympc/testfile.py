from numpy import linalg
import mpc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

A = np.array([[-.1, -.4], [.4, -.1]])
B = np.array([[.5, .01], [1, .01]]) * .1
C = np.eye(2)
Q = np.eye(2)
R = np.eye(2) * .1
upper_lim = 100
ulim = np.array([[-upper_lim, -upper_lim],[1, 1]])
lqrcontroller = np.array([[1.398, 5.068],[.044, .0299]])

def linsys(x, u):
    state = x[:2]
    measure = x[2:]
    
    statedot = np.dot(A, state) + np.dot(B , u)
    return statedot

x_0 = np.array([[1], [1.1]])
x_controlled_hist = [x_0]
x_hist = [x_0]
u_hist = []
u_lqr_hist = []
iters = 100
N = 10
lqrcontroller = lqrcontroller

x = x_0 + 0
x_controlled = x_0 + 0
for i in range(iters):
    lin, quad, constr, lims = mpc.mpc_to_qp(x_controlled, A, B, C, Q, R, ulim, N)
    u_raw = mpc.solve_qp_ip(quad, lin.reshape(N * 2 + 2), constr[:N * 2 + 2,:N * 2 + 2], lims[:N * 2 + 2,].reshape(N * 2 + 2))
    u = u_raw[:2].reshape(2,1)
    dx_controlled = linsys(x_controlled, u)
    u_lqr = - lqrcontroller @ x
    u_lqr[u_lqr > upper_lim] = upper_lim
    dx = linsys(x,u_lqr)
    x = x + dx
    x_controlled = x_controlled + dx_controlled
    print("Done a loop")
    x_hist.append(x)
    x_controlled_hist.append(x_controlled)
    u_hist.append(u)
    u_lqr_hist.append(u_lqr)


x_hist_disp = np.array(x_hist).reshape(iters+1, 2)
x_controlled_hist_disp = np.array(x_controlled_hist).reshape(iters+1, 2)
u_hist_disp = np.array(u_hist).reshape(iters, 2)
u_lqr_hist_disp = np.array(u_lqr_hist).reshape(iters, 2)


plt.plot(u_lqr_hist_disp, label="LQR")
plt.plot(u_hist_disp, label="MPC")
plt.legend()
plt.figure(2)
plt.plot(x_hist_disp[:,0], x_hist_disp[:,1], label="LQR")
plt.plot(x_controlled_hist_disp[:,0], x_controlled_hist_disp[:,1], label="MPC")
plt.legend()
plt.show()