# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec
from consav.linear_interp import interp_1d

@nb.njit
def solve_hh_backwards(par,z_trans,y,rl,vbeg_l_plus,vbeg_l,l,c):
    """ solve backwards with vbeg_l_a from previous iteration """

    # solve
    for i_fix in range(par.Nfix):

        # a. solution step
        for i_z in range(par.Nz):

            e = par.z_grid[i_z]  # productivity
            ye = y*e

            # ii. inverse foc
            c_endo = (par.beta_grid[i_fix] * vbeg_l_plus[i_fix, i_z, :]) ** (-1 / par.sigma)
            m_endo = c_endo + par.l_grid

            for i_l in range(par.Nl):

                # ii. interpolation to fixed grid

                m = (1 + rl) * par.l_grid[i_l] + ye

                l[i_fix,i_z,i_l] = interp_1d(m_endo, par.l_grid, m)

                l[i_fix,i_z,i_l] = np.fmax(l[i_fix,i_z,i_l],0.0)  # enforce borrowing constraint
                c[i_fix,i_z,i_l] = m-l[i_fix, i_z,i_l]
                c[i_fix,i_z,i_l] = np.fmax(c[i_fix,i_z,i_l],0.0)    # enforce non-negative consumption

        # b. expectation step
        v_l = (1+rl)*c[i_fix]**(-par.sigma)

        for i_z in range(par.Nz):
            for i_l in range(par.Nl):
                vbeg_l[i_fix,i_z,i_l] = 0.0
                for i_z_plus in range(par.Nz):
                    vbeg_l[i_fix, i_z, i_l] += z_trans[i_fix,i_z,i_z_plus]*v_l[i_z_plus,i_l]