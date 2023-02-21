import time
import numpy as np
from scipy import optimize

from consav import elapsed
from consav.grids import equilogspace
from consav.markov import log_rouwenhorst


def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    # a. beta
    par.beta_grid[:] = np.linspace(par.beta_mean - par.beta_delta, par.beta_mean + par.beta_delta, par.Nfix)

    # b. l
    par.l_grid[:] = equilogspace(par.l_min, par.l_max, par.Nl)

    # c. e
    sigma = np.sqrt(par.sigma_e ** 2 * (1 - par.rho_e ** 2))
    par.z_grid[:], ss.z_trans[0, :, :], e_ergodic, _, _ = log_rouwenhorst(par.rho_e, sigma, n=par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    # start with single point distribution for each z
    for i_fix in range(par.Nfix):
        ss.Dz[i_fix, :] = e_ergodic / par.Nfix
        for i_z in range(par.Nz):
            ss.Dbeg[i_fix, i_z, :] *= 0.0
            ss.Dbeg[i_fix, i_z, 0] = ss.Dz[i_fix, i_z]

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_l = np.zeros((par.Nfix, par.Nz, par.Nl))
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):
            e = par.z_grid[i_z]
            y = ss.y
            ye = y * e
            m = (1 + ss.rl) * par.l_grid + ye
            c = m
            v_l[i_fix, i_z, :] = (1 + ss.rl) * c ** (-par.sigma)

            ss.vbeg_l[i_fix, :, :] = ss.z_trans[i_fix] @ v_l[i_fix, :, :]


def evaluate_ss(model, do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    ss.y = par.Z_target
    ss.r = par.r_ss_target
    ss.rl = par.r_ss_target - par.xi
    ss.ey = 0.0

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print, Dbeg=ss.Dbeg)

    model._compute_jac_hh()

    # MPC_1_year = model.jac_hh[('C_hh', 'y')][0, 0:4].sum()
    # MPC_1_year = model.jac_hh[('C_hh', 'ey')][0, 0:4].sum()
    MPC_1_year = np.sum([model.jac_hh[('C_hh', 'ey')][i, 0] / (1 + ss.r) ** i for i in range(4)])

    # MPCs_model = [model.jac_hh[('C_hh', 'y')][0, (t * 4):(t * 4) + 4].sum() for t in [0, 1, 2, 3, 4, 5]]

    # j. clearing
    # ss.MPC_match = (np.array(par.MPC_target) - np.array(MPCs_model)).sum()
    ss.MPC_match = par.MPC_target - MPC_1_year



def objective_ss(x, model, do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta_mean = x[0]
    evaluate_ss(model, do_print=do_print)

    return ss.MPC_match


def find_ss(model, do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    if do_print: print('Find optimal beta for market clearing')

    t0 = time.time()
    res = optimize.root(objective_ss, par.beta_mean, method='hybr', tol=par.tol_ss, args=(model))   # method: hybr

    assert res["success"], res["message"]

    # b. final evaluation
    if do_print: print('final evaluation')
    objective_ss([par.beta_mean], model, do_print=do_print)

    # check targets
    if do_print:
        print(f'steady state found in {elapsed(t0)}')
        print(f' beta   = {par.beta_mean:6.4}')
        print(f'Discrepancy in annual MPC = {ss.MPC_match:12.8f}')

