from collections import defaultdict
import numpy as np
from scipy.sparse.linalg import norm
from time import time
from scipy import sparse
import datetime


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    x_k = x_0

    if trace:
        history = {'time': [], 'func' : [], 'duality_gap' : []}
        if max(x_0.shape) <= 2:
            history['x'] = []
    else:
        history = None

    if display:
        print("Debug info")

    x_opt = x_0
    phi_opt = oracle.func(x_0)
    phi_k = phi_opt
    start = time()
    for k in range(max_iter+1):
        dg = oracle.duality_gap(x_k)

        if trace:
            current_time = time()
            history['time'].append(current_time - start)
            history['func'].append(phi_k)
            history['duality_gap'].append(dg)
            if max(x_0.shape) <= 2:
                history['x'].append(np.copy(x_k))

        if dg <= tolerance:    # TODO: Check the correctness
            return x_k, 'success', history

        sg_k = oracle.subgrad(x_k).T
        a_k = alpha_0/np.sqrt(k+1)
        div = np.sqrt(sg_k.dot(sg_k.T))
        if sparse.issparse(div):
            div = div[0, 0]
        x_k = x_k - a_k*sg_k/div
        phi_k = oracle.func(x_k)

        if phi_k <= phi_opt:
            phi_opt = phi_k
            x_opt = x_k

    return x_opt, 'iterations_exceeded', history




def proximal_gradient_descent(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Proximal gradient descent for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    x_k = x_0

    if trace:
        history = {'time': [], 'func' : [], 'duality_gap' : [], 'inner_iters': []}
        if max(x_0.shape) <= 2:
            history['x'] = []
    else:
        history = None

    if display:
        print("Debug info")

    start = time()
    L = L_0
    inner_iter_counter = 0
    for k in range(max_iter):

        dg = oracle.duality_gap(x_k)
        g_k = oracle.grad(x_k)
        f_k = oracle._f.func(x_k)
        phi_k = f_k + oracle._h.func(x_k)

        if trace:
            current_time = time()
            history['time'].append(current_time - start)
            history['func'].append(phi_k)
            history['duality_gap'].append(dg)
            history['inner_iters'].append(inner_iter_counter)
            if max(x_0.shape) <= 2:
                history['x'].append(np.copy(x_k))

        if dg <= tolerance:    # TODO: Check the correctness
            return x_k, 'success', history


        while True:
            inner_iter_counter += 1
            y = oracle.prox(x_k - 1/L*g_k.T, 1/L)
            y_xk = y - x_k
            rhs = y_xk.dot(g_k) + L/2*y_xk.dot(y_xk.T)
            if sparse.issparse(rhs):
                rhs = rhs[0, 0]
            rhs += f_k
            if oracle._f.func(y) <= rhs:
                break
            L *= 2
        x_k = y
        L = max(L_0, L/2)

    dg = oracle.duality_gap(x_k)

    if trace:
        current_time = time()
        history['time'].append(current_time - start)
        history['func'].append(oracle.func(x_k))
        history['duality_gap'].append(dg)
        history['inner_iters'].append(inner_iter_counter)
        if max(x_0.shape) <= 2:
            history['x'].append(np.copy(x_k))


    return x_k, 'iterations_exceeded', history


def accelerated_proximal_gradient_descent(oracle, x_0, L_0=1.0, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Proximal gradient descent for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values phi(y_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
    """

    if trace:
        history = {'time': [], 'func' : [], 'duality_gap' : [], 'inner_iters': []}
    else:
        history = None

    if display:
        print("Debug info")

    L = L_0
    A = 0
    y_k = x_0
    v_k = x_0
    x_opt = x_0
    phi_opt = oracle.func(x_0)
    phi_y = oracle.func(y_k)
    af = 0
    start = time()
    inner_iters_counter = 0
    for k in range(max_iter):

        dg = oracle.duality_gap(y_k)

        if trace:
            current_time = time()
            history['time'].append(current_time - start)
            history['func'].append(phi_y)
            history['inner_iters'].append(inner_iters_counter)
            history['duality_gap'].append(dg)

        if dg < tolerance:
            return x_opt, "success", history

        while True:
            inner_iters_counter += 1
            a = (1+np.sqrt(1 + 4*L*A))/(2*L)
            tau = a/(a+A)
            z = tau*v_k + (1 - tau)*y_k
            g_z = oracle.grad(z)
            y = oracle.prox(z - 1/L*g_z.T, 1/L)
            f_y = oracle._f.func(y)
            f_z = oracle._f.func(z)
            phi_y = f_y + oracle._h.func(y)
            phi_z = f_z + oracle._h.func(z)
            if phi_y < phi_opt:
                x_opt = y
                phi_opt = phi_y
            if phi_z < phi_opt:
                x_opt = z
                phi_opt = phi_z
            y_z = y - z
            if sparse.issparse(y):
                rhs = f_z + (g_z.T.dot(y_z.T) + L/2*y_z.dot(y_z.T))[0, 0]
            else:
                rhs = f_z + g_z.dot(y_z) + L/2*y_z.dot(y_z.T)
            if f_y <= rhs:
                break
            L *= 2

        y_k = y
        A += a
        af += a*g_z
        v_k = oracle.prox(x_0.T - af, A)
        L = max(L_0, L/2)

    dg = oracle.duality_gap(y_k)

    if trace:
        current_time = time()
        history['time'].append(current_time - start)
        history['func'].append(oracle.func(y_k))
        history['inner_iters'].append(inner_iters_counter)
        history['duality_gap'].append(dg)

    return x_opt, "iterations_exceeded", history