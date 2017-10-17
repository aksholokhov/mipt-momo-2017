from time import time

import numpy as np

from task2.utils import get_line_search_tool


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    x_k = np.copy(x_0).astype("float64")
    if max_iter is None:
        max_iter = x_k.shape[0]

    if trace:
        history = {'time': [], 'residual_norm' : []}
        if len(x_0) <= 2:
            history['x'] = []
    else:
        history = None

    if display:
        print("Debug info")

    g_k = matvec(x_0) - b
    d_k = -g_k

    start = time()
    for k in range(max_iter + 1):

        if trace:
            current_time = time()
            history['time'].append(current_time - start)
            history['residual_norm'].append(np.linalg.norm(g_k))
            if len(x_0) <= 2:
                history['x'].append(np.copy(x_k))

        if np.linalg.norm(g_k) <= tolerance*np.linalg.norm(b):
            return x_k, 'success', history

        Adk = matvec(d_k)
        a = g_k.dot(g_k)/Adk.dot(d_k)

        x_k += a*d_k
        g_k_plus_1 = g_k + a*Adk

        d_k = -g_k_plus_1 + g_k_plus_1.dot(g_k_plus_1)/g_k.dot(g_k)*d_k
        g_k = g_k_plus_1

    if trace:
        current_time = time()
        history['time'].append(current_time - start)
        history['residual_norm'].append(np.linalg.norm(g_k))
        if len(x_0) <= 2:
            history['x'].append(np.copy(x_k))

    return x_k, 'iterations_exceeded', history

def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """



    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    if trace:
        history = {'time': [], 'func': [], 'grad_norm': []}
        if len(x_0) <= 2:
            history['x'] = []
    else:
        history = None

    if display:
        print("Debug info")

    g_0 = oracle.grad(x_0)

    start = time()
    lbfgs_history = []

    x_old = None
    g_old = None

    for k in range(max_iter + 1):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)

        if np.isnan(f_k).any() or np.isnan(g_k).any() or np.isinf(f_k).any() or np.isinf(g_k).any():
            return x_k, "computational_error", history

        if x_old is not None and g_old is not None:
            lbfgs_history.append((x_k - x_old, g_k - g_old))
            if len(lbfgs_history) > memory_size:
                lbfgs_history = lbfgs_history[1:]

        if np.linalg.norm(g_k)**2 <= tolerance * np.linalg.norm(g_0)**2:       # TODO: may be wrong
            if trace:
                current_time = time()
                history['time'].append(current_time - start)
                history['func'].append(f_k)
                history['grad_norm'].append(np.linalg.norm(g_k))
                if len(x_0) <= 2:
                    history['x'].append(np.copy(x_k))
            return x_k, 'success', history


        d_k = lbfgs_direction(g_k, lbfgs_history)

        a_k = line_search_tool.line_search(oracle, x_k, d_k)

        if trace:
            current_time = time()
            history['time'].append(current_time - start)
            history['func'].append(f_k)
            history['grad_norm'].append(np.linalg.norm(g_k))
            if len(x_0) <= 2:
                history['x'].append(np.copy(x_k))

        x_old = np.copy(x_k)
        g_old = np.copy(g_k)
        x_k = x_k + a_k * d_k


    return x_k, 'iterations_exceeded', history

def lbfgs_direction(g_k, lbfgs_history):
    if len(lbfgs_history) == 0:
        return -g_k

    s, y = lbfgs_history[-1]
    gamma_0 = y.dot(s)/y.dot(y)
    return lbfgs_multiply(-g_k, lbfgs_history, gamma_0)

def lbfgs_multiply(v, lbfgs_history, gamma_0):
    if len(lbfgs_history) == 0:
        return gamma_0*v
    (s, y), lbfgs_history_rest = lbfgs_history[-1], lbfgs_history[:-1]
    u = v - s.dot(v)/y.dot(s)*y
    z = lbfgs_multiply(u, lbfgs_history_rest, gamma_0)
    return z + (s.dot(v) - y.dot(z))/y.dot(s)*s

def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    if trace:
        history = {'time': [], 'func': [], 'grad_norm': []}
        if len(x_0) <= 2:
            history['x'] = []
    else:
        history = None

    if display:
        print("Debug info")

    g_0 = oracle.grad(x_0)

    start = time()
    for k in range(max_iter + 1):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)

        if np.isnan(f_k).any() or np.isnan(g_k).any() or np.isinf(f_k).any() or np.isinf(g_k).any():
            return x_k, "computational_error", history

        if np.linalg.norm(g_k)**2 <= tolerance * np.linalg.norm(g_0)**2:       # TODO: may be wrong
            if trace:
                current_time = time()
                history['time'].append(current_time - start)
                history['func'].append(f_k)
                history['grad_norm'].append(np.linalg.norm(g_k))
                if len(x_0) <= 2:
                    history['x'].append(np.copy(x_k))
            return x_k, 'success', history

        solution_found = False

        while not solution_found:
            n_k = min(0.5, np.sqrt(np.linalg.norm(g_k)))
            d_k, message, _ = conjugate_gradients(lambda v: oracle.hess_vec(x_k, v), -g_k, -g_k,
                                                  tolerance = n_k)
            if message != "success" or g_k.dot(d_k) >= 0:
                n_k /= 10
                continue

            solution_found = True

        a_k = line_search_tool.line_search(oracle, x_k, d_k)

        if trace:
            current_time = time()
            history['time'].append(current_time - start)
            history['func'].append(f_k)
            history['grad_norm'].append(np.linalg.norm(g_k))
            if len(x_0) <= 2:
                history['x'].append(np.copy(x_k))

        x_k = x_k + a_k * d_k

    return x_k, 'iterations_exceeded', history