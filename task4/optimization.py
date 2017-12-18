from collections import defaultdict
import numpy as np
from scipy.sparse.linalg import norm
from time import time
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

    start = time()
    for k in range(max_iter+1):
        dg = oracle.duality_gap(x_k)

        if trace:
            current_time = time()
            history['time'].append(current_time - start)
            history['func'].append(oracle.func(x_k))
            history['duality_gap'].append(dg)
            if max(x_0.shape) <= 2:
                history['x'].append(np.copy(x_k))

        if dg <= tolerance:    # TODO: Check the correctness
            return x_k, 'success', history

        sg_k = oracle.subgrad(x_k)
        a_k = alpha_0/np.sqrt(k+1)
        x_k = x_k - a_k*sg_k/np.sqrt(sg_k.dot(sg_k.T))

    return x_k, 'iterations_exceeded', history




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
        history = {'time': [], 'func' : [], 'duality_gap' : []}
        if max(x_0.shape) <= 2:
            history['x'] = []
    else:
        history = None

    if display:
        print("Debug info")

    start = time()
    L = L_0
    for k in range(max_iter):

        dg = oracle.duality_gap(x_k)

        if trace:
            current_time = time()
            history['time'].append(current_time - start)
            history['func'].append(oracle.func(x_k))
            history['duality_gap'].append(dg)
            if max(x_0.shape) <= 2:
                history['x'].append(np.copy(x_k))

        if dg <= tolerance:    # TODO: Check the correctness
            return x_k, 'success', history

        g_k = oracle.grad(x_k)
        f_k = oracle._f.func(x_k)
        while True:
            y = oracle.prox(x_k - 1/L*g_k, 1/L)
            y_xk = y - x_k
            if oracle._f.func(y) <= f_k + g_k.dot(y_xk.T) + L/2*y_xk.dot(y_xk.T):
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
        history = {'time': [], 'func' : [], 'duality_gap' : []}
    else:
        history = None

    if display:
        print("Debug info")

    L = L_0
    A = 0
    y = x_0
    v = x_0
    x_opt = x_0
    phi_opt = oracle.func(x_0)
    af = 0
    start = time()
    for k in range(max_iter+1):
        while True:
            if k == 1:
                a = 1/L
            else:
                a = 1+np.sqrt(1 + 4*L*A)/(2*L)
            tau = a/(a+A)
            z = tau*v + (1 - tau)*y
            y = oracle.prox(z - 1/L*oracle.grad(z), 1/L)
            f_y = oracle._f.func(y)
            f_z = oracle._f.func(z)
            g_z = oracle.grad(z)
            phi_y = f_y + oracle._h.func(y)
            phi_z = f_z + oracle._h.func(z)
            if phi_y < phi_opt:
                x_opt = y
                phi_opt = phi_y
            if phi_z < phi_opt:
                x_opt = z
                phi_opt = phi_z
            y_z = y - z
            if f_y <= f_z + g_z.dot((y - z).T) + L/2*y_z.dot(y_z.T)**2:
                break
            L /= 2

        dg = oracle.duality_gap(y)

        if trace:
            current_time = time()
            history['time'].append(current_time - start)
            history['func'].append(phi_y)
            history['duality_gap'].append(dg)

        if dg < tolerance:
            return x_opt, "success", history

        A = A + a
        af += a*g_z
        v = oracle.prox(x_0 - af, 1)
        L = max(L_0, L/2)

    return x_opt, "iterations_exceeded", history