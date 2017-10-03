import numpy as np
import scipy as sp
from scipy import sparse
from scipy.special import expit


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


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.matvec_b_adam_A = lambda x: sparse.diags(b, 0).dot(x) # thanx to Anton Zakharenkov for an idea of a fast implementation of Adamar's multiplication
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        return np.mean(
                np.vectorize(lambda x: np.logaddexp(0, x))(self.matvec_b_adam_A(self.matvec_Ax(x)) * (-1))
            ) + self.regcoef / 2 * np.dot(x, x)

    def grad(self, x):
        m = self.b.shape[0]

        return - (1 / m) * self.matvec_ATx(
                            self.matvec_b_adam_A(
                                np.vectorize(expit)(-1*self.matvec_b_adam_A(self.matvec_Ax(x))
                                )
                            )
                        )  + self.regcoef * x

    def hess(self, x):
        m = self.b.shape[0]

        return (1 / m) * self.matmat_ATsA(
                    np.vectorize(lambda x: expit(x) * (1 - expit(x)))(
                        -1 * self.matvec_b_adam_A(self.matvec_Ax(x))
                    )
                )  + self.regcoef * np.eye(len(x))


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)
    matmat_ATsA = lambda x: A.T.dot(sp.sparse.diags(x).dot(A))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    return np.array([(func(x + eps * e) - func(x)) / eps for e in np.eye(x.shape[0])])

def hess_finite_diff(func, x, eps=1e-5):
    return np.array([[(func(x + eps * e_i + eps * e_j)
                               - func(x + eps * e_i)
                               - func(x + eps * e_j)
                               + func(x)) / eps**2
                      for e_j in np.eye(x.shape[0])] for e_i in np.eye(x.shape[0])])
