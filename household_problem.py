# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec
from consav.linear_interp import interp_1d

@nb.njit
def solve_hh_backwards(par,z_trans,tau,wN,ez,eg_transfer,ra,rl,vbeg_l_a_plus,vbeg_l_a,l,c,a,uce):
    """ solve backwards with vbeg_l_a from previous iteration """

    # unpack
    A_target = par.A_target
    ra_ss = par.r_ss_target


    # solve
    for i_fix in range(par.Nfix):

        # a. solution step
        for i_z in range(par.Nz):

            e = par.z_grid[i_z]  # productivity
            Z = (1 - tau) * wN + ez
            Ze = Z*e + eg_transfer  # labor income

            # ii. inverse foc
            for i_a in range(par.Na):

                # ii. inverse foc
                c_endo = (par.beta_grid[i_fix] * vbeg_l_a_plus[i_fix, i_z, :, i_a]) ** (-1 / par.sigma)
                m_endo = c_endo + par.l_grid

                for i_l in range(par.Nl):

                    a_lag = par.a_grid[i_a]

                    # ii. interpolation to fixed grid
                    d = ra_ss/(1+ra_ss)*(1+ra)*a_lag + par.chi*((1+ra)*a_lag-(1+ra_ss)*A_target)
                    m = (1 + rl) * par.l_grid[i_l] + Ze + d

                    l[i_fix,i_z,i_l,i_a] = interp_1d(m_endo, par.l_grid, m)

                    l[i_fix,i_z,i_l,i_a] = np.fmax(l[i_fix,i_z,i_l,i_a],0.0)  # enforce borrowing constraint
                    c[i_fix,i_z,i_l,i_a] = m-l[i_fix, i_z,i_l,i_a]
                    #  for some combinations of l,e and d, m can get negative because distribution to illiquid account
                    #  (expenses) > income.
                    c[i_fix,i_z,i_l,i_a] = np.fmax(c[i_fix,i_z,i_l,i_a],0.0)    # enforce non-negative consumption
                    a[i_fix,i_z,i_l,i_a] = (1+ra)*a_lag-d  # next periods illiquid assets
                    assert not a[i_fix,i_z,i_l,i_a] < 0, 'negative illiquid assets'

                    uce[i_fix,i_z,i_l,i_a] = e*c[i_fix,i_z,i_l,i_a]**(-par.sigma) # productivity weighted marg. util.

        # b. expectation step
        v_l_a = (1+rl)*c[i_fix]**(-par.sigma)

        for i_z in range(par.Nz):
            for i_l in range(par.Nl):
                for i_a in range(par.Na):
                    vbeg_l_a[i_fix,i_z,i_l,i_a] = 0.0
                    for i_z_plus in range(par.Nz):
                        vbeg_l_a[i_fix, i_z, i_l, i_a] += z_trans[i_fix,i_z,i_z_plus]*v_l_a[i_z_plus,i_l,i_a]