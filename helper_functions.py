# helper functions that extend the tools in the GEModelClass package

import time
from copy import deepcopy
import numpy as np
import numba as nb

# from EconModel import jit
# from consav.misc import elapsed
# from consav.linear_interp import binary_search
#
# from GEModelTools.simulate_hh import find_i_and_w_1d_1d
# from GEModelTools.simulate_hh import find_i_and_w_1d_1d_path


@nb.njit
def integrate_marg_util(c, D, z_grid, sigma):
    """ integrate marginal utility multiplied by productivity """

    assert c.ndim == 4 and D.ndim == 4, 'c and D do not have the fitting dimensions'
    assert c.shape == D.shape, 'c and D do not have the same dimensions'

    Nfix = c.shape[0]
    Ne = c.shape[1]
    Nl = c.shape[2]
    Na = c.shape[3]

    int_val = 0.0

    for i_fix in nb.prange(Nfix):
        for i_e in nb.prange(Ne):
            e_i = z_grid[i_e]
            for i_a in nb.prange(Na):
                for i_l in nb.prange(Nl):
                    int_val = int_val + e_i * c[i_fix, i_e, i_l, i_a]**(-sigma) * D[i_fix, i_e, i_l, i_a]

    return int_val


# could not get rid of
# "TypingError: Failed in nopython mode pipeline (step: convert make_function into JIT functions)
# Cannot capture the non-constant value associated with variable 'Y' in a function that will escape."
# "AssertionError: Failed in nopython mode pipeline (step: inline calls to locally defined closures)"
# TODO: remove comment
# @nb.njit
def broyden_solver_cust(f, x0, kwargs_dict=None, jac=None,
                        tol=1e-8, max_iter=100, backtrack_fac=0.5, max_backtrack=30,
                        do_print=False):
    """ numerical solver using the broyden method """

    # a. initial
    x = x0.ravel()
    y = f(x, kwargs_dict)

    if len(x) < len(y) and do_print:
        print("Dimension of x, is less than dimension of y."
              " Using least-squares criterion to solve for approximate root.")

    # b. iterate
    for it in range(max_iter):

        # i. current difference
        abs_diff = np.max(np.abs(y))
        if do_print:
            print('iteration; max. abs. error')
            print(it, abs_diff)
            # print(' it = {:3d} -> max. abs. error = {:8.2e}'.format(it, abs_diff))

        if abs_diff < tol: return x

        # # init jac not neccessary and run into numba problems
        # if not isinstance(jac, np.ndarray):
        #     # initialize J with Newton!
        #     if jac == None and kwargs_dict == None:
        #         raise NotImplementedError('residual function needs kwargs_dict at the moment!')
        #     elif jac == None:
        #         jac = obtain_J(f, x, y, kwargs_dict)

        # ii. new x
        if len(x) == len(y):
            dx = np.linalg.solve(jac, -y)
        elif len(x) < len(y):
            dx = np.linalg.lstsq(jac, -y, rcond=None)[0]
        else:
            raise ValueError("Dimension of x is greater than dimension of y."
                             " Cannot solve underdetermined system.")

        # iii. evalute with backtrack
        for _ in range(max_backtrack):

            try:  # evaluate
                ynew = f(x + dx, kwargs_dict)
                if np.any(np.isnan(ynew)): raise ValueError('found nan value')
            except Exception:  # .. as e:
                if do_print: print('backtracking...')
                dx *= backtrack_fac
            else:  # update jac and break from backtracking
                dy = ynew - y
                jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx) ** 2), dx)
                y = ynew
                x += dx
                break

        else:

            raise ValueError('GEModelTools: Number of backtracks exceeded')

    else:

        raise ValueError('GEModelTools: No convergence of broyden solver in solving for investment')

# TODO: remove comment
# @nb.njit
def obtain_J(f, x, y, kwargs_dict=None, h=1E-5):
    """Finds Jacobian f'(x) around y=f(x)"""
    nx = x.shape[0]
    ny = y.shape[0]
    J = np.empty((ny, nx))

    for i in range(nx):
        dx = h * (np.arange(nx) == i)
        J[:, i] = (f(x + dx, kwargs_dict) - y) / h
    return J

# taken from sequence-jacobian repository
def residual_with_linear_continuation(residual, bounds, kwargs_dict=None, eval_at_boundary=False,
                                      boundary_epsilon=1e-4, penalty_scale=1e1,
                                      verbose=False):
    """Modify a residual function to implement bounds by an additive penalty for exceeding the boundaries
    provided, scaled by the amount the guess exceeds the boundary.

    e.g. For residual function f(x), desiring x in (0, 1) (so assuming eval_at_boundary = False)
         If the guess for x is 1.1 then we will censor to x_censored = 1 - boundary_epsilon, and return
         f(x_censored) + penalty (where the penalty does not require re-evaluating f() which may be costly)

    residual: `function`
        The function whose roots we want to solve for
    bounds: `dict`
        A dict mapping the names of the unknowns (`str`) to length two tuples corresponding to the lower and upper
        bounds.
    eval_at_boundary: `bool`
        Whether to allow the residual function to be evaluated at exactly the boundary values or not.
        Think of it as whether the solver will treat the bounds as creating a closed or open set for the search space.
    boundary_epsilon: `float`
        The amount to adjust the proposed guess, x, by to calculate the censored value of the residual function,
        when the proposed guess exceeds the boundaries.
    penalty_scale: `float`
        The linear scaling factor for adjusting the penalty for the proposed unknown values exceeding the boundary.
    verbose: `bool`
        Whether to print out additional information for how the constrained residual function is behaving during
        optimization. Useful for tuning the solver.
    """
    lbs = np.asarray([v[0] for v in bounds.values()])
    ubs = np.asarray([v[1] for v in bounds.values()])

    def constr_residual(x, kwargs_dict=None, residual_cache=[]):
        """Implements a constrained residual function, where any attempts to evaluate x outside of the
        bounds provided will result in a linear penalty function scaled by `penalty_scale`.

        Note: We are purposefully using residual_cache as a mutable default argument to cache the most recent
        valid evaluation (maintain state between function calls) of the residual function to induce solvers
        to backstep if they encounter a region of the search space that returns nan values.
        See Hitchhiker's Guide to Python post on Mutable Default Arguments: "When the Gotcha Isn't a Gotcha"
        """
        if eval_at_boundary:
            x_censored = np.where(x < lbs, lbs, x)
            x_censored = np.where(x > ubs, ubs, x_censored)
        else:
            x_censored = np.where(x < lbs, lbs + boundary_epsilon, x)
            x_censored = np.where(x > ubs, ubs - boundary_epsilon, x_censored)

        residual_censored = residual(x_censored, kwargs_dict)

        if verbose:
            print(f"Attempted x is {x}")
            print(f"Censored x is {x_censored}")
            print(f"The residual_censored is {residual_censored}")

        if np.any(np.isnan(residual_censored)):
            # Provide a scaled penalty to the solver when trying to evaluate residual() in an undefined region
            residual_censored = residual_cache[0] * penalty_scale

            if verbose:
                print(f"The new residual_censored is {residual_censored}")
        else:
            if not residual_cache:
                residual_cache.append(residual_censored)
            else:
                residual_cache[0] = residual_censored

        if verbose:
            print(f"The residual_cache is {residual_cache[0]}")

        # Provide an additive, scaled penalty to the solver when trying to evaluate residual() outside of the boundary
        residual_with_boundary_penalty = residual_censored + \
                                         (x - x_censored) * penalty_scale * residual_censored
        return residual_with_boundary_penalty

    return constr_residual