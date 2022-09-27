# functions that replace the tools in the GEModelClass package

import time
from copy import deepcopy
import numpy as np
import numba as nb

# from EconModel import jit
from consav.misc import elapsed
from consav.linear_interp import binary_search

class no_jit():
    """ class to replace jit() class if no numba compilation is wanted"""
    def __init__(self,model,show_exc=True):
        """ load namespace references """
        self.model = model
        self.show_exc = show_exc
        for ns in model.namespaces:
            setattr(self,ns,getattr(model,ns))

    def __enter__(self):
        return self.model

    def __exit__(self, exc_type, exc_value, tb):
        model = self.model
        for ns in model.namespaces:
            normal = getattr(self,ns)
            setattr(model,ns,normal)

# to counter circular import
@nb.njit(parallel=True)
def find_i_and_w_1d_1d(pol1, grid1, i, w):
    """ find indices and weights for simulation """

    Nfix = pol1.shape[0]
    Nz = pol1.shape[1]
    Nendo1 = pol1.shape[2]

    for i_z in nb.prange(Nz):
        for i_fix in nb.prange(Nfix):
            for i_endo in nb.prange(Nendo1):
                # a. policy
                pol1_ = pol1[i_fix, i_z, i_endo]

                # b. find i_ such a_grid[i_] <= a_ < a_grid[i_+1]
                i_ = i[i_fix, i_z, i_endo] = binary_search(0, grid1.size, grid1, pol1_)

                # c. weight
                w[i_fix, i_z, i_endo] = (grid1[i_ + 1] - pol1_) / (grid1[i_ + 1] - grid1[i_])

                # d. avoid extrapolation
                w[i_fix, i_z, i_endo] = np.fmin(w[i_fix, i_z, i_endo], 1.0)
                w[i_fix, i_z, i_endo] = np.fmax(w[i_fix, i_z, i_endo], 0.0)

# to counter circular import
@nb.njit
def find_i_and_w_1d_1d_path(T, path_pol1, grid1, path_i, path_w):
    """ find indices and weights for simulation along transition path"""

    for k in range(T):
        t = (T - 1) - k

        find_i_and_w_1d_1d(path_pol1[t], grid1, path_i[t], path_w[t])

    return k

@nb.njit(parallel=True)
def find_i_and_w_2d_1d(pol1, grid_endo, grid1, grid2, i, w):
    """

    find indices and weights for simulation

    :param pol1: policy function defined over a grid of 2 variables
    :param grid_endo: (1d array) grid of relevant policy variable. Has to be the same os one of the grids.
    :param grid1: (1d array) grid of relevant policy variable
    :param grid2: (1d array) grid of second policy variable
    :param i: ss_dict['pol_indices']
    :param w: ss_dict['pol_weights']
        """

    Nfix = pol1.shape[0]
    Nz = pol1.shape[1]
    Nendo1 = grid1.size
    Nendo2 = grid2.size
    # Nendo = pol1.shape[2]
    # Nexo = pol1.shape[3]

    for i_z in nb.prange(Nz):
        for i_fix in nb.prange(Nfix):
            for i_endo1 in nb.prange(Nendo1):
                for i_endo2 in nb.prange(Nendo2):
                    # a. policy
                    pol1_ = pol1[i_fix, i_z, i_endo1, i_endo2]

                    # b. find i_ such l_grid[i_] <= l_ < l_grid[i_+1] given l_grid[i_endo]
                    i_ = i[i_fix, i_z, i_endo1, i_endo2] = binary_search(0, grid_endo.size, grid_endo, pol1_)

                    # c. weight
                    w[i_fix, i_z, i_endo1, i_endo2] = (grid_endo[i_ + 1] - pol1_) / (grid_endo[i_ + 1] - grid_endo[i_])

                    # d. avoid extrapolation
                    w[i_fix, i_z, i_endo1, i_endo2] = np.fmin(w[i_fix, i_z, i_endo1, i_endo2], 1.0)
                    w[i_fix, i_z, i_endo1, i_endo2] = np.fmax(w[i_fix, i_z, i_endo1, i_endo2], 0.0)


def _find_i_and_w_dict(model,ss_dict):
    """ find policy indices and weights from dict """

    par = model.par

    if len(model.grids_hh) == 1:
        pol1 = ss_dict[f'{model.grids_hh[0]}']
        grid1 = getattr(par,f'{model.grids_hh[0]}_grid')
        find_i_and_w_1d_1d(pol1,grid1,ss_dict['pol_indices'],ss_dict['pol_weights'])
    elif len(model.grids_hh) == 2 and len(model.pols_hh) == 1:
        # two variables (l, a) that determine the grid of the policy function (l')
        pol1 = ss_dict[f'{model.grids_hh[0]}']
        grid1 = getattr(par, f'{model.grids_hh[0]}_grid')
        grid2 = getattr(par, f'{model.grids_hh[1]}_grid')
        find_i_and_w_2d_1d(pol1, grid1, grid1, grid2, ss_dict['pol_indices'], ss_dict['pol_weights'])
    elif len(model.grids_hh) == 2 and len(model.pols_hh) == 2:
        # two variables (l, a) that determine the grid of the two policy function (l', a')
        pol1 = ss_dict[f'{model.grids_hh[0]}']
        pol2 = ss_dict[f'{model.grids_hh[1]}']
        grid1 = getattr(par, f'{model.grids_hh[0]}_grid')
        grid2 = getattr(par, f'{model.grids_hh[1]}_grid')
        find_i_and_w_2d_1d(pol1, grid1, grid1, grid2, ss_dict['pol_indices'][0], ss_dict['pol_weights'][0])
        find_i_and_w_2d_1d(pol2, grid2, grid1, grid2, ss_dict['pol_indices'][1], ss_dict['pol_weights'][1])
    else:
        raise NotImplementedError

@nb.njit
def find_i_and_w_2d_1d_path(T, path_pol1, grid1, grid2, path_i, path_w):
    """ find indices and weights for simulation along transition path"""

    for k in range(T):
        t = (T-1)-k
        find_i_and_w_2d_1d(path_pol1[t], grid1, grid1, grid2, path_i[t], path_w[t])
    return k

def _find_i_and_w_path(model):
    """ find indices and weights along the transition path"""

    par = model.par
    path = model.path

    if len(model.grids_hh) == 1:
        path_pol1 = getattr(path,f'{model.grids_hh[0]}')
        grid1 = getattr(par,f'{model.grids_hh[0]}_grid')
        find_i_and_w_1d_1d_path(par.T, path_pol1, grid1, path.pol_indices, path.pol_weights)
    elif len(model.grids_hh) == 2 and len(model.pols_hh) == 1:
        # two variables (l, a) that determine the grid of the policy function (l')
        path_pol1 = getattr(path, f'{model.grids_hh[0]}')
        grid1 = getattr(par, f'{model.grids_hh[0]}_grid')
        grid2 = getattr(par, f'{model.grids_hh[1]}_grid')
        find_i_and_w_2d_1d_path(par.T, path_pol1, grid1, grid2, path.pol_indices, path.pol_weights)
    elif len(model.grids_hh) == 2 and len(model.pols_hh) == 2:
        # two variables (l, a) that determine the grid of the two policy function (l', a')
        path_pol1 = getattr(path, f'{model.grids_hh[0]}')
        path_pol2 = getattr(path, f'{model.grids_hh[1]}')
        grid1 = getattr(par, f'{model.grids_hh[0]}_grid')
        grid2 = getattr(par, f'{model.grids_hh[1]}_grid')
        find_i_and_w_2d_1d_path(par.T, path_pol1, grid1, grid2, path.pol_indices[:, 0], path.pol_weights[:, 0])
        find_i_and_w_2d_1d_path(par.T, path_pol2, grid1, grid2, path.pol_indices[:, 1], path.pol_weights[:, 1])
    else:
        raise NotImplemented


@nb.njit(parallel=True)
def simulate_hh_forwards_exo(Dbeg, z_trans_T, D):
    """ exogenous stochastic transition given transition matrix """

    Nfix = Dbeg.shape[0]

    if Dbeg.ndim < 4:  # dimensions equal to 2 or 3
        for i_fix in nb.prange(Nfix):
            D[i_fix] = z_trans_T[i_fix] @ Dbeg[i_fix]
    elif Dbeg.ndim == 4:  # more dimensions than: Nfix, Nz and Na
        Nz = Dbeg.shape[1]
        Nl = Dbeg.shape[2]
        Na = Dbeg.shape[3]
        # for i_fix in nb.prange(Nfix):
        #     for i_a in nb.prange(Na):
        #         D[i_fix,:,:,i_a] = z_trans_T[i_fix] @ Dbeg[i_fix,:,:,i_a]
        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
                for i_l in nb.prange(Nl):
                    for i_a in nb.prange(Na):
                        D[i_fix, i_z, i_l, i_a] = 0.0
                        for i_z_plus in nb.prange(Nz):
                            D[i_fix, i_z, i_l, i_a] += z_trans_T[i_fix, i_z, i_z_plus] * Dbeg[i_fix, i_z_plus , i_l, i_a]
    else:
        raise NotImplementedError

# @nb.njit(parallel=True)
# def simulate_hh_forwards_exo_transpose(Dbeg,z_trans,D):
#     """ simulate given indices and weights """
#     Nfix = Dbeg.shape[0]
#     if Dbeg.ndim < 4:  # dimensions
#         for i_fix in nb.prange(Nfix):
#             D[i_fix] = z_trans[i_fix]@Dbeg[i_fix]
#     elif Dbeg.ndim == 4:  # more dimensions than: Nfix, Nz and Na
#         Nz = Dbeg.shape[1]
#         Nl = Dbeg.shape[2]
#         Na = Dbeg.shape[3]
#         D = np.zeros_like(Dbeg)
#         for i_fix in nb.prange(Nfix):
#             for i_l in nb.prange(Nl):
#                 for i_a in nb.prange(Na):
#                     for i_z in nb.prange(Nz):
#                         for i_z_plus in nb.prange(Nz):
#                             D[i_fix, i_z, i_l, i_a] += z_trans[i_fix, i_z, i_z_plus] * Dbeg[i_fix, i_z_plus, i_l, i_a]
#     else:
#         raise NotImplementedError

@nb.njit(parallel=True)
def simulate_hh_forwards_exo_transpose(Dbeg,z_trans):
    """ simulate given indices and weights """
    Nfix = Dbeg.shape[0]
    D = np.zeros_like(Dbeg)
    if Dbeg.ndim < 4:  # dimensions
        for i_fix in nb.prange(Nfix):
            D[i_fix] = z_trans[i_fix]@Dbeg[i_fix]
    elif Dbeg.ndim == 4:  # more dimensions than: Nfix, Nz and Na
        Nz = Dbeg.shape[1]
        Nl = Dbeg.shape[2]
        Na = Dbeg.shape[3]
        for i_fix in nb.prange(Nfix):
            for i_l in nb.prange(Nl):
                for i_a in nb.prange(Na):
                    for i_z in nb.prange(Nz):
                        for i_z_plus in nb.prange(Nz):
                            D[i_fix, i_z, i_l, i_a] += z_trans[i_fix, i_z, i_z_plus] * Dbeg[i_fix, i_z_plus, i_l, i_a]
    else:
        raise NotImplementedError
    return D


@nb.njit(parallel=True)
def simulate_hh_forwards_endo_1d(D, i, w, Dbeg_plus):
    """ forward simulation with 1d distribution """
    Nfix = D.shape[0]
    Nz = D.shape[1]
    Nendo1 = D.shape[2]

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):

            Dbeg_plus[i_fix, i_z, :] = 0.0
            for i_endo in range(Nendo1):
                # i. from
                D_ = D[i_fix, i_z, i_endo]

                # ii. to
                i_ = i[i_fix, i_z, i_endo]
                w_ = w[i_fix, i_z, i_endo]
                Dbeg_plus[i_fix, i_z, i_] += D_ * w_
                Dbeg_plus[i_fix, i_z, i_ + 1] += D_ * (1.0 - w_)


@nb.njit(parallel=True)
def simulate_hh_forwards_endo_2d_1iw(D, i, w, Dbeg_plus):
    """ forward simulation with 2d distribution but only along one grid dimension """
    Nfix = D.shape[0]
    Nz = D.shape[1]
    Nendo1 = D.shape[2]
    Nendo2 = D.shape[3]

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            Dbeg_plus[i_fix, i_z, :, :] = 0.0
            for i_endo2 in nb.prange(Nendo2):
                for i_endo1 in nb.prange(Nendo1):
                    # i. from
                    D_ = D[i_fix, i_z, i_endo1, i_endo2]

                    # ii. to
                    i_ = i[i_fix, i_z, i_endo1, i_endo2]
                    w_ = w[i_fix, i_z, i_endo1, i_endo2]
                    Dbeg_plus[i_fix, i_z, i_, i_endo2] += D_ * w_
                    Dbeg_plus[i_fix, i_z, i_ + 1, i_endo2] += D_ * (1.0 - w_)


@nb.njit(parallel=True)
def simulate_hh_forwards_endo_2d_2iw(D, i, w, Dbeg_plus):
    """ forward simulation with 2d distribution along both grid dimension """
    Nfix = D.shape[0]
    Nz = D.shape[1]
    Nendo1 = D.shape[2]
    Nendo2 = D.shape[3]

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            Dbeg_plus[i_fix, i_z, :, :] = 0.0
            for i_endo1 in nb.prange(Nendo1):
                for i_endo2 in nb.prange(Nendo2):
                    # i. from
                    D_ = D[i_fix, i_z, i_endo1, i_endo2]

                    # ii. to
                    i_1_ = i[0, i_fix, i_z, i_endo1, i_endo2]
                    i_2_ = i[1, i_fix, i_z, i_endo1, i_endo2]
                    w_1_ = w[0, i_fix, i_z, i_endo1, i_endo2]
                    w_2_ = w[1, i_fix, i_z, i_endo1, i_endo2]

                    Dbeg_plus[i_fix, i_z, i_1_, i_2_] += w_1_ * w_2_ * D_
                    Dbeg_plus[i_fix, i_z, i_1_ + 1, i_2_] += (1 - w_1_) * w_2_ * D_
                    Dbeg_plus[i_fix, i_z, i_1_, i_2_ + 1] += w_1_ * (1 - w_2_) * D_
                    Dbeg_plus[i_fix, i_z, i_1_ + 1, i_2_ + 1] += (1 - w_1_) * (1 - w_2_) * D_
    # return Dbeg_plus



@nb.njit
def simulate_hh_forwards_endo(D, i, w, Dbeg_plus):
    """ replaced function to simulate endougenous deterministic transition given indices and weights """

    Ndim_i = i.ndim
    Ndim_D = D.ndim

    if Ndim_D == 3:
        simulate_hh_forwards_endo_1d(D, i, w, Dbeg_plus)
    elif Ndim_D == 4 and Ndim_i == Ndim_D:
        simulate_hh_forwards_endo_2d_1iw(D, i, w, Dbeg_plus)
    elif Ndim_D == 4 and Ndim_D + 1 == Ndim_i:
        simulate_hh_forwards_endo_2d_2iw(D, i, w, Dbeg_plus)
    else:
        raise NotImplementedError

@nb.njit(parallel=True)
def simulate_hh_forwards_endo_1d_trans(Dbeg_plus, i, w):
    """ forward simulation with 1d distribution """

    Nfix = Dbeg_plus.shape[0]
    Nz = Dbeg_plus.shape[1]
    Nendo1 = Dbeg_plus.shape[2]
    D = np.zeros_like(Dbeg_plus)
    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            for i_endo in range(Nendo1):
                i_ = i[i_fix, i_z, i_endo]
                w_ = w[i_fix, i_z, i_endo]
                D[i_fix, i_z, i_endo] = w_ * Dbeg_plus[i_fix, i_z, i_] + (1.0 - w_) * Dbeg_plus[i_fix, i_z, i_ + 1]
    return D


@nb.njit(parallel=True)
def simulate_hh_forwards_endo_2d_1iw_trans(Dbeg_plus, i, w):
    """ forward simulation with 2d distribution but only along one grid dimension """
    Nfix = Dbeg_plus.shape[0]
    Nz = Dbeg_plus.shape[1]
    Nendo1 = Dbeg_plus.shape[2]
    Nendo2 = Dbeg_plus.shape[3]
    D = np.zeros_like(Dbeg_plus)
    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            for i_endo2 in nb.prange(Nendo2):
                for i_endo1 in nb.prange(Nendo1):
                    i_ = i[i_fix, i_z, i_endo1, i_endo2]
                    w_ = w[i_fix, i_z, i_endo1, i_endo2]
                    D[i_fix, i_z, i_endo1, i_endo2] = w_ * Dbeg_plus[i_fix, i_z, i_, i_endo2] + \
                                                      (1.0 - w_) * Dbeg_plus[i_fix, i_z, i_ + 1, i_endo2]
    return D

@nb.njit(parallel=True)
def simulate_hh_forwards_endo_2d_2iw_trans(Dbeg_plus, i, w):
    """ forward simulation with 2d distribution along both grid dimension """
    Nfix = Dbeg_plus.shape[0]
    Nz = Dbeg_plus.shape[1]
    Nendo1 = Dbeg_plus.shape[2]
    Nendo2 = Dbeg_plus.shape[3]
    D = np.zeros_like(Dbeg_plus)
    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            for i_endo1 in nb.prange(Nendo1):
                for i_endo2 in nb.prange(Nendo2):
                    i_1_ = i[0, i_fix, i_z, i_endo1, i_endo2]
                    i_2_ = i[1, i_fix, i_z, i_endo1, i_endo2]
                    w_1_ = w[0, i_fix, i_z, i_endo1, i_endo2]
                    w_2_ = w[1, i_fix, i_z, i_endo1, i_endo2]

                    D[i_fix, i_z, i_endo1, i_endo2] = ((w_1_ * w_2_) * Dbeg_plus[i_fix, i_z, i_endo1, i_endo2] + \
                                                      + w_1_ * (1 - w_2_) * Dbeg_plus[i_fix, i_z, i_endo1, i_endo2 + 1] + \
                                                      (1 - w_1_) * w_2_ * Dbeg_plus[i_fix, i_z, i_endo1 + 1, i_endo2] + \
                                                      (1 - w_1_) * (1 - w_2_) * Dbeg_plus[i_fix, i_z, i_endo1 + 1, i_endo2 + 1])
    return D

# @nb.njit
# def simulate_hh_forwards_endo_transpose(Dbeg_plus, i, w, D):
#     """ simulate given indices and weights """
#
#     Ndim_i = i.ndim
#     Ndim_D = Dbeg_plus.ndim
#
#     if Ndim_D == 3:
#         simulate_hh_forwards_endo_1d_trans(Dbeg_plus, i, w, D)
#     elif Ndim_D == 4 and Ndim_i == Ndim_D:
#         simulate_hh_forwards_endo_2d_1iw_trans(Dbeg_plus, i, w, D)
#     elif Ndim_D == 4 and Ndim_D + 1 == Ndim_i:
#         simulate_hh_forwards_endo_2d_2iw_trans(Dbeg_plus, i, w, D)
#     else:
#         raise NotImplementedError

@nb.njit
def simulate_hh_forwards_endo_transpose(Dbeg_plus, i, w):
    """ simulate given indices and weights """

    Ndim_i = i.ndim
    Ndim_D = Dbeg_plus.ndim

    if Ndim_D == 3:
        D = simulate_hh_forwards_endo_1d_trans(Dbeg_plus, i, w)
    elif Ndim_D == 4 and Ndim_i == Ndim_D:
        D = simulate_hh_forwards_endo_2d_1iw_trans(Dbeg_plus, i, w)
    elif Ndim_D == 4 and Ndim_D + 1 == Ndim_i:
        D = simulate_hh_forwards_endo_2d_2iw_trans(Dbeg_plus, i, w)
    else:
        raise NotImplementedError
    return D


