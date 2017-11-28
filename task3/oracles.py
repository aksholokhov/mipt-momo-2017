import numpy as np
import scipy
from scipy.special import expit


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    mu = min(1, regcoef/max(ATAx_b))*Ax_b
    return 1/2*Ax_b.dot(Ax_b.T) + regcoef*scipy.linalg.norm(x, ord=1) + 1/2*mu.dot(mu.T) + b.dot(mu.T)
