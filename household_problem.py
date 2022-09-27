# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec


@nb.njit
def solve_hh_backwards(par, z_trans, Z, ra, rl, vbeg_l_a_plus,vbeg_l_a, l, c, a):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    A_target = (par.hh_wealth_Y_ratio-par.L_Y_ratio)    # TODO: add *ss.Y
    r_ss = par.r_ss_target

    # i. prepare
    # grid for transfer of illiquid account
    da = lambda a: r_ss / (1 + r_ss) * (1+ra)*a + par.chi * (
            (1 + ra) * a - (1 + r_ss) * A_target)  # distribution from illiquid account
    dA_grid = np.array([da(a_i) for a_i in par.a_grid])

    for i_fix in range(par.Nfix):

        # a. solution step

        for i_z in range(par.Nz):

            # i. prepare
            e = par.z_grid[i_z]
            Ze = Z * e  # labor income

            # ii. inverse foc
            for i_a, a_i in enumerate(par.a_grid):
                # ii. inverse foc
                c_endo = (par.beta_grid[i_fix]*vbeg_l_a_plus[i_fix, i_z, :, i_a])**(-1/par.sigma)
                m_endo = c_endo + par.l_grid

                # ii. interpolation to fixed grid
                da_i = da(a_i)
                y = Ze + da_i   # income from labor and illiquid assets
                m = (1+rl)*par.l_grid + y

                interp_1d_vec(m_endo, par.l_grid, m, l[i_fix, i_z, :, i_a])
                l[i_fix, i_z, :, i_a] = np.fmax(l[i_fix, i_z, :, i_a], 0.0)  # enforce borrowing constraint
                c[i_fix, i_z, :, i_a] = m - l[i_fix, i_z, :, i_a]
                a[i_fix, i_z, :, i_a] = (1 + ra) * a_i - da(a_i)    # next periods illiquid assets

                # uce[i_fix, i_z, :, i_a] = e * c[i_fix, i_z, :, i_a] ** (-par.sigma) # productivity weighted marg. util.

        # b. expectation step
        v_l_a = (1 + rl) * c[i_fix] ** (-par.sigma)
        for i_z in range(par.Nz):
            for i_l in range(par.Nl):
                for i_a in range(par.Na):
                    vbeg_l_a[i_fix, i_z, i_l, i_a] = 0.0
                    for i_z_plus in range(par.Nz):
                        vbeg_l_a[i_fix, i_z, i_l, i_a] += z_trans[i_fix, i_z, i_z_plus] * v_l_a[i_z_plus , i_l, i_a]

        # alternative, but full looping faster than matmult on non-contiguous array or using np.ascontiguousarray()
        # for i_a in range(par.Na):
        #     vbeg_l_a[i_fix, :, :, i_a] = z_trans[i_fix] @ v_l_a[ :, :, i_a]