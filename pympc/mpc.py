import numpy as np



def solve_lqr_ip(A: np.array, B: np.array, C:np.array, lambda: float,  ):
    """ Source: Algorithm 1 in A Microcontroller Implementation
    of Constrained Model Predictive Control by Abbes et al."""
    