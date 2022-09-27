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
@nb.njit
def broyden_solver_cust(f, x0, kwargs_dict=None, jac=None,
                        tol=1e-8, max_iter=100, backtrack_fac=0.5, max_backtrack=30,
                        do_print=False):
    """ numerical solver using the broyden method """

    # a. initial
    x = x0.ravel()
    y = f(x, kwargs_dict)

    if len(x) < len(y):

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

        # # init jac of neccessary
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
            dx = np.linalg.lstsq(jac, -y)[0]
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

@nb.njit
def obtain_J(f, x, y, kwargs_dict=None, h=1E-5):
    """Finds Jacobian f'(x) around y=f(x)"""
    nx = x.shape[0]
    ny = y.shape[0]
    J = np.empty((ny, nx))

    for i in range(nx):
        dx = h * (np.arange(nx) == i)
        J[:, i] = (f(x + dx, kwargs_dict) - y) / h
    return J


# def broyden_solver_cust(f, x0, jac=None,
#                    tol=1e-8, max_iter=100, backtrack_fac=0.5, max_backtrack=30,
#                    do_print=False, do_print_unknowns=False, model=None,
#                    fixed_jac=False):
#     """ numerical solver using the broyden method """
#
#     # a. initial
#     x = x0.ravel()
#     y = f(x)
#
#     if len(x) < len(y):
#         print(f"Dimension of x, {len(x)} is less than dimension of y, {len(y)}."
#               f" Using least-squares criterion to solve for approximate root.")
#
#     # b. iterate
#     for it in range(max_iter):
#
#         # i. current difference
#         abs_diff = np.max(np.abs(y))
#         if do_print:
#
#             print(f' it = {it:3d} -> max. abs. error = {abs_diff:8.2e}')
#
#             if not model is None and do_print_unknowns:
#                 for unknown in model.unknowns:
#                     minval = np.min(model.path.__dict__[unknown][0, :])
#                     meanval = np.mean(model.path.__dict__[unknown][0, :])
#                     maxval = np.max(model.path.__dict__[unknown][0, :])
#                     print(f'   {unknown:15s}: {minval = :7.2f} {meanval = :7.2f} {maxval = :7.2f}')
#
#             if not model is None and len(model.targets) > 1:
#                 y_ = y.reshape((len(model.targets), -1))
#                 for i, target in enumerate(model.targets):
#                     print(f'   {np.max(np.abs(y_[i])):8.2e} in {target}')
#
#         if abs_diff < tol: return x
#
#         if not isinstance(jac, np.ndarray):
#             if jac == None:
#                 # initialize J with Newton!
#                 jac = obtain_J(f, x, y)
#
#         # ii. new x
#         if len(x) == len(y):
#             # print(f'J:{jac}')
#             # print(f'y:{y}')
#             dx = np.linalg.solve(jac, -y)
#             # print(dx)
#         elif len(x) < len(y):
#             dx = np.linalg.lstsq(jac, -y, rcond=None)[0]
#         else:
#             raise ValueError(f"Dimension of x, {len(x)} is greater than dimension of y, {len(y)}."
#                              f" Cannot solve underdetermined system.")
#
#         # iii. evalute with backtrack
#         for _ in range(max_backtrack):
#
#             try:  # evaluate
#                 ynew = f(x + dx)
#                 if np.any(np.isnan(ynew)): raise ValueError('found nan value')
#             except Exception as e:  # backtrack
#                 if do_print: print(f'backtracking...')
#                 dx *= backtrack_fac
#             else:  # update jac and break from backtracking
#                 dy = ynew - y
#                 if not fixed_jac:
#                     # print(f'jac{jac.shape} \n '
#                     #       f'dy: {dy.shape} \n'
#                     #       f'dx: {dx.shape} \n'
#                     #       f'norm(dx): {np.linalg.norm(dx).shape}')
#                     jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx) ** 2), dx)
#                 y = ynew
#                 x += dx
#                 break
#
#         else:
#
#             raise ValueError(f'GEModelTools: Number of backtracks exceeds {max_backtrack}')
#
#     else:
#
#         raise ValueError(f'GEModelTools: No convergence after {max_iter} iterations with broyden_solver(tol={tol:.1e})')
#
# def obtain_J(f, x, y, h=1E-5):
#     """Finds Jacobian f'(x) around y=f(x)"""
#     nx = x.shape[0]
#     ny = y.shape[0]
#     J = np.empty((ny, nx))
#
#     for i in range(nx):
#         dx = h * (np.arange(nx) == i)
#         J[:, i] = (f(x + dx) - y) / h
#     return J



