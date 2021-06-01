import numpy as np



def solve_lqr_ip(A: np.array, B: np.array, C:np.array, lambd: float):
    """ Source: 
        Slides 7 - 13 of https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-323-principles-of-optimal-control-spring-2008/lecture-notes/lec16.pdf
    """
    return 0

def mpc_to_qp(x, A, B, C, Q, R, N):
    """
    Sources:
        Slides 7 - 13 of https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-323-principles-of-optimal-control-spring-2008/lecture-notes/lec16.pdf
        https://github.com/pronenewbits/Arduino_Constrained_MPC_Library

    A: n_x by n_x
    B: n_x by n_u
    C: n_y by n_x
    G: (N+1) * n_y by n_x
    H: (N+1) * n_y by n_u * (N+1)
    Z: N+1 * n_y by 1
    U: N+1 * n_u by 1
    W1: N+1 * n_y by N+1 * n_y+1
    W2: N+1 * n_u by N+1 * n_u
    C: 2 * ((N+1) * n_u), (N+1) * n_u)
    (u_m = 2 * n_u by 1)
    """
    n_x = A.shape[0]
    n_y = C.shape[0]
    n_u = B.shape[1]
    G = np.zeros(((N+1) * n_y, n_x))
    H = np.zeros(((N+1) * n_y, n_u * (N+1)))
    for i in range(N+1):
        G[n_y * i:n_y * (i+1), : ] = C @ A**i
        for j in range(i):
            h_res = C @ A ** (i-(j+1)) @ B
            H[n_y * i:n_y * (i+1), n_u * j : n_u * (j+1)] = h_res

    W_1 = np.zeros(((N+1) * n_y, (N+1) * n_y))
    W_2 = np.zeros(((N+1) * n_u, (N+1) * n_u))
    for i in range(N+1):
        for j in range(N+1):
            W_1[n_y * i :n_y * (i+1), n_y * j : n_y * (j+1)] = Q
            W_1[n_u * i :n_u * (i+1), n_u * j : n_u * (j+1)] = R

    #H_1 = G.T @ W_1 @ G
    H_2 = 2 * (np.dot(x.T, G.T) @ W_1 @ H)
    H_3 = 2 * (H.T @ W_1 @ H + W_2)

    C = np.zeros((2 * ((N+1) * n_u), (N+1) * n_u))
    for d in range(N+1):
        C[n_u * d :n_u * (d+1), n_u * d : n_u * (d+1)] = - np.eye(n_u)
        C[n_u * (d + N + 1) :n_u * ((d+N + 1)+1), n_u * d : n_u * (d+1)] = np.eye(n_u)
    return H_2.T, H_3, C

def solve_qp_ip(B, b, A, d, tol = 1e-2, sigma = .1):
    """ Solves the quadratic problem Outlined in source 2 using the interior point method.
    Sources: 
        Algorithm 1 in A Microcontroller Implementation of Constrained Model Predictive Control by Abbes et al.
        3.6 in https://www.math.uh.edu/~rohop/fall_06/Chapter3.pdf
    """
    n = B.shape[0]
    x0 = np.zeros(n)
    x_old = np.ones(n) * tol * 10
    z0 = d - A @ x0
    mu0 = np.ones(n)
    e = np.ones(n)
    x_list = []
    z_list  = []
    mu_list  = []

    x = x0
    z = z0
    mu = mu0
    while(np.linalg.norm(x_old - x) > tol):
        print("Iteration")
        D_mu = np.diag(mu)
        dmuinv = np.linalg.inv(D_mu)
        Z = np.diag(z)
        rb = B @ x + A.T@ mu - b
        rd = A @ x + z - d
        kappa = np.dot(z, mu) / n
        g = - Z @ D_mu @ e + e * sigma * kappa 

        leftblock = np.zeros((2 * n, 2 * n))
        leftblock[0:n,0:n] = B
        leftblock[n:,0:n] = A
        leftblock[0:n,n:] = A.T
        leftblock[n:,n:] = - dmuinv @ Z

        rightvect = np.zeros((2*n))
        rightvect[0:n] = -rb
        rightvect[n:] = -rd - dmuinv @ g

        dxdmu = np.linalg.solve(leftblock, rightvect)
        dx = dxdmu[:n]
        dmu = dxdmu[n:]
        dz = dmuinv @ (g - Z @ dmu)

        alpha = calculate_alpha_ip(x, z, mu, dx, dz, dmu)

        x_old = x + 0
        x = x + alpha * dx
        z = z + alpha * dz
        mu = mu + alpha * dmu
    return x

def calculate_alpha_ip(x, z, mu, dx, dz, dmu):
    alphas = np.concatenate((-x/dx, -z/dz, -mu/dmu))
    alphas = alphas[alphas > 0]
    alpha1 = np.min(alphas)
    return np.min([alpha1 * .995, 1])

"""
#Testing mpc to qp
x = np.zeros(3)
A = np.eye(3)
B = np.array([[1, .2], [.5, .3], [.1, .4]])
C = np.eye(3)
Q = np.eye(3)
R = np.eye(2)
N = 2
print(mpc_to_qp(x, A, B, C, Q, R, N))
"""

"""
#Testing interior point solver
B = np.array([[14, 0, 4, 6], [0, 1, 0, 0], [4, 0, 5, 6], [6, 0, 6, 10]])
b = - 1 *  np.ones(4)
A = np.eye(4)
d = np.ones(4) * 10

print(solve_qp_ip(B, b, A, d))
"""
"""
#Testing alpha calculation outlined in Algorithm 1 in A Microcontroller Implementation
x = np.ones(3)
z = np.ones(3)
mu = np.ones(3)
dx = np.ones(3) * -.1
dz = np.ones(3) * -2
dmu = np.ones(3) * .1
print(calculate_alpha_ip(x, z, mu, dx, dz, dmu))
"""