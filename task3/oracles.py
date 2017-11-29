import numpy as np
import scipy
from scipy.special import expit

def lasso_duality_gap(x, Ax_b, ATAx_b, b, reg_coef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    mu = min(1, reg_coef / max(ATAx_b)) * Ax_b
    return 1/2*Ax_b.dot(Ax_b.T) + reg_coef * scipy.linalg.norm(x, ord=1) + 1 / 2 * mu.dot(mu.T) + b.dot(mu.T)


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

class LassoBarrierOracle(BaseSmoothOracle):

    def __init__(self, A, b, reg_coef, t, ATb = None, ATA = None):
        self.__A = A
        self.__AT = A.T
        self.__b = b
        if ATb is None:
            self.__ATb = self.__AT.dot(b)
        else:
            self.__ATb = ATb
        if ATA is None:
            self.__ATA = self.__AT.dot(A)
        else:
            self.__ATA = ATA
        self.__n = max(b.shape)
        self.__regcoef = reg_coef
        self.__t = t

    @classmethod
    def fromOther(cls, other, t):
        return cls(other.__A, other.__b, other.__regcoef, t, other.__ATb, other.__ATA)

    def func(self, y):
        n = self.__n
        x = y[:n], u = y[n+1:]
        Ax_b = self.__A.dot(x.T) - self.__b
        return self.__t*(1/2*Ax_b.dot(Ax_b.T) +  self.__regcoef*u.sum())\
               + sum([np.log(x_i + u_i) + np.log(u_i - x_i) for x_i, u_i in zip(x, u)])

    def grad(self, y):
        n = self.__n
        x = y[:n], u = y[n + 1:]
        res = np.zeros(2*n)
        res[:n] = self.__t*(self.__ATA.dot(x.T) - self.__ATb) \
                  + np.array([1/(x_i + u_i) - 1/(u_i - x_i) for x_i, u_i in zip(x, u)])
        res[n+1:] = np.array([1/(x_i + u_i) + 1/(u_i - x_i) for x_i, u_i in zip(x, u)]) + self.__t*self.__regcoef
        return res

    def hess(self, y):
        n = self.__n
        x = y[:n], u = y[n + 1:]
        res = np.zeros((2*n, 2*n))
        res[:n, :n] = self.__t*self.__ATA + np.diag(np.array([-1/(x_i + u_i)**2 - 1/(u_i - x_i)**2 for x_i, u_i in zip(x, u)]))
        res[n+1:, :n] = np.diag(np.array([-1/(x_i + u_i)**2 + 1/(u_i - x_i)**2 for x_i, u_i in zip(x, u)]))
        res[:n, n+1] = res[n+1:, :n]
        res[n+1:, n+1:] = np.diag(np.array([-1/(x_i + u_i)**2 - 1/(u_i - x_i)**2 for x_i, u_i in zip(x, u)]))
        return res