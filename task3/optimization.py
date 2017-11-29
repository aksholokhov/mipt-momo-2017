from collections import defaultdict
import numpy as np
from scipy.linalg import norm, solve, cholesky, cho_solve, cho_factor
from time import time
import datetime
from utils.utils import get_line_search_tool
from task3.oracles import LassoBarrierOracle

def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5, 
                         tolerance_inner=1e-8, max_iter=100, 
                         max_iter_inner=20, t_0=1, gamma=10, 
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """


    n = max(x_0.shape)

    def ldg(x):
        Ax_b = A.dot(x[:n]) - b
        ATAx_b = A.T.dot(Ax_b)
        return lasso_duality_gap(x[:n], Ax_b, ATAx_b, b, reg_coef)

    x_k = np.concatenate([x_0, u_0])

    if trace:
        history = {'time': [], 'func': [], "duality_gap" : []}
        if len(x_0) <= 2:
            history['x'] = []
    else:
        history = None

    if display:
        print("Debug info")

    t = t_0
    oracle = LassoBarrierOracle(A, b, reg_coef, t)

    start = time()

    q = np.zeros((2*n, 2*n))
    for i in range(n):
        q[i, i] = 1
        q[i, i + n] = -1
        q[i + n, i] = -1
        q[i + n, i + n] = -1

    for k in range(max_iter + 1):
        x_new, message, inner_history = barrier_lasso_subroutine_solver(oracle, x_k,
                                                                  max_iter=max_iter_inner,
                                                                  tolerance=tolerance_inner,
                                                                  line_search_options={
                                                                      'method': 'Armijo',
                                                                      'c1' : c1,
                                                                      'alpha_0' : 1
                                                                  },
                                                                  cons = q,
                                                                  ldg = ldg,
                                                                  init_time = time())

        cur_ldg = ldg(x_new)

        if trace:
            Ax_b = A.dot(x_new[:n]) - b
            current_time = time()
            history['time'] += inner_history["time"]
            history['time'].append(current_time)
            history['func'] += inner_history["func"]
            history['func'].append(Ax_b.dot(Ax_b.T) + reg_coef * np.linalg.norm(x_new[:n], ord=1))
            history['duality_gap'] += inner_history['duality_gap']
            history["duality_gap"].append(cur_ldg)
            if len(x_0) <= 2:
                history['x'].append(np.copy(x_k))

        if cur_ldg <= tolerance:
            history['time'] = np.array(history["time"]) - start
            return (x_k[:n], x_k[n:]), 'success', history

        t = t * gamma
        oracle = LassoBarrierOracle.fromOther(oracle, t)
        x_k = x_new

    history['time'] = np.array(history["time"]) - start
    return (x_k[:n], x_k[n:]), 'iterations_exceeded', history



def barrier_lasso_subroutine_solver(oracle, x_0, trace = True, tolerance=1e-5, max_iter=100,
           line_search_options=None, cons = None, ldg = None, init_time = 0):

    x_k = np.copy(x_0)
    n = int(max(x_k.shape)/2)

    if trace:
        history = {'time': [], 'func': [], 'duality_gap': []}
    else:
        history = None

    g_0 = oracle.grad(x_0)

    start = time()
    for k in range(max_iter + 1):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        h_k = oracle.hess(x_k)

        if np.linalg.norm(g_k) ** 2 <= tolerance * np.linalg.norm(g_0) ** 2:
            if trace:
                current_time = time()
                history['time'].append(current_time)
                history['func'].append(f_k)
                history['duality_gap'].append(ldg(x_k))
            return x_k, 'success', history

        try:
            U = cholesky(h_k, check_finite=True)
            d_k = cho_solve((U, False), -g_k, check_finite=True)
        except np.linalg.LinAlgError as e:
            if "not positive" in str(e):
                return x_k, "newton_direction_error", history
            else:
                return x_k, "computational_error", history

        I = [i for i, q in enumerate(cons) if q.dot(d_k) > 0]
        if len(I) != 0:
            alpha_max = 0.99*min([-cons[i].dot(x_k.T)/(cons[i].dot(d_k.T)) for i in I])
        else:
            alpha_max = 1
        line_search_options["alpha_0"] = min(1, alpha_max)
        line_search_tool = get_line_search_tool(line_search_options)
        a_k = line_search_tool.line_search(oracle, x_k, d_k)

        if trace:
            current_time = time()
            history['time'].append(current_time)
            history['func'].append(f_k)
            history['duality_gap'].append(ldg(x_k))

        x_k = x_k + a_k * d_k

    return x_k, 'iterations_exceeded', history
