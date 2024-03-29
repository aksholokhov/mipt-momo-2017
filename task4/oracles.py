import numpy as np
import scipy
from scipy.special import expit
from scipy import sparse

class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class BaseNonsmoothConvexOracle(object):
    """
    Base class for implementation of oracle for nonsmooth convex function.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        """
        Computes arbitrary subgradient vector at point x.
        """
        raise NotImplementedError('Subgrad is not implemented.')

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    def __init__(self, matvec_Ax, matvec_ATx, b):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b

    def func(self, x):
        Ax_b = self.matvec_Ax(x.T).T - self.b
        if sparse.issparse(Ax_b):
            res = 0.5 * Ax_b.dot(Ax_b.T)[0, 0]
        else:
            res = 0.5 * Ax_b.dot(Ax_b.T)
        return res

    def grad(self, x):
        return self.matvec_ATx((self.matvec_Ax(x.T).T - self.b).T)


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef*abs(x).sum()

    def prox(self, x, alpha):
        def soft_threshold(c, alpha):
            if c > alpha:
                return c - alpha
            elif c < -alpha:
                return c + alpha
            else:
                return 0

        if sparse.issparse(x):
            p = np.squeeze(x.toarray())
        else:
            p = x
        res = np.array([soft_threshold(e, self.regcoef*alpha) for e in p])
        if sparse.issparse(x):
            return sparse.csr_matrix(res)
        return res


class LassoProxOracle(BaseCompositeOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    def duality_gap(self, x):
        Ax_b = self._f.matvec_Ax(x.T)
        Ax_b -= self._f.b.T
        ATAx_b = self._f.matvec_ATx(Ax_b)

        mu = min(1, self._h.regcoef / abs(ATAx_b).max()) * Ax_b
        res = 1 / 2 * Ax_b.T.dot(Ax_b)
        res += 1 / 2 * mu.T.dot(mu)
        res += self._f.b.dot(mu)
        if sparse.issparse(res):
            res = res[0, 0]
        return res + self._h.regcoef * abs(x).sum()


class LassoNonsmoothOracle(BaseNonsmoothConvexOracle):
    """
    Oracle for nonsmooth convex function
        0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    def __init__(self, matvec_Ax, matvec_ATx, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax_b = self.matvec_Ax(x.T).T - self.b
        res = 0.5 * Ax_b.dot(Ax_b.T)
        if sparse.issparse(res):
            res = res[0, 0]
        return res + self.regcoef*abs(x).sum()

    def subgrad(self, x):
        # sbg = np.array([e if e != 0 else 1 for e in self.regcoef * np.sign(np.squeeze(x.toarray()))])
        # if sparse.issparse(x):
        #     sbg = sparse.csr_matrix(sbg)

        sbg = self.regcoef*x.sign().T if sparse.issparse(x) else self.regcoef*np.sign(x)
        return self.matvec_ATx((self.matvec_Ax(x.T).T - self.b).T) + sbg

    def duality_gap(self, x):
        Ax_b = self.matvec_Ax(x.T)
        Ax_b -= self.b.T
        ATAx_b = self.matvec_ATx(Ax_b)

        mu = min(1, self.regcoef / abs(ATAx_b).max()) * Ax_b
        res = 1 / 2 * Ax_b.T.dot(Ax_b)
        res += 1 / 2 * mu.T.dot(mu)
        res += self.b.dot(mu)
        if sparse.issparse(res):
            res = res[0, 0]
        return res + self.regcoef * abs(x).sum()


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    mu = min(1, regcoef / abs(ATAx_b).max()) * Ax_b
    return 1 / 2 * Ax_b.dot(Ax_b.T) + regcoef * abs(x).sum() + 1 / 2 * mu.dot(mu.T) + b.dot(mu.T)
    

def create_lasso_prox_oracle(A, b, regcoef):
    AT = A.T
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: AT.dot(x)
    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))


def create_lasso_nonsmooth_oracle(A, b, regcoef):
    AT = A.T
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: AT.dot(x)
    return LassoNonsmoothOracle(matvec_Ax, matvec_ATx, b, regcoef)

