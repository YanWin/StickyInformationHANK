import time
import numpy as np
from scipy import optimize

from consav import elapsed

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

from helper_functions import integrate_marg_util
from GEModelTools.replaced_functions import simulate_hh_forwards_endo



def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    # a. beta
    par.beta_grid[:] = np.linspace(par.beta_mean - par.beta_delta, par.beta_mean + par.beta_delta, par.Nfix)


    # b. a, l
    par.a_grid[:] = equilogspace(par.a_min, par.a_max, par.Na)
    par.l_grid[:] = equilogspace(par.l_min, par.l_max, par.Nl)

    # c. e
    sigma = np.sqrt(par.sigma_e ** 2 * (1 - par.rho_e ** 2))
    par.z_grid[:], ss.z_trans[0, :, :], e_ergodic, _, _ = log_rouwenhorst(par.rho_e, sigma, n=par.Ne)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    # init Dbeg
    # start with single point distribution for each z
    for i_fix in range(par.Nfix):
        ss.Dz[i_fix, :] = e_ergodic / par.Nfix
        for i_z in range(par.Nz):
            ss.Dbeg[i_fix, i_z, :, :] *= 0.0
            ss.Dbeg[i_fix, i_z, 0, 0] = ss.Dz[i_fix, i_z]

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_l_a = np.zeros((par.Nfix, par.Nz, par.Nl, par.Na))

    da = lambda a: a * ss.ra  # distribution from illiquid account
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):
            e = par.z_grid[i_z]
            Z = (1-ss.tau)*ss.w*ss.N
            Ze = Z*e
            for i_a, a_i in enumerate(par.a_grid):
                y = Ze + da(a_i)
                m = (1 + ss.rl) * par.l_grid + y
                c = m
                v_l_a[i_fix, i_z, :, i_a] = (1 + ss.rl) * c ** (-par.sigma)  # foc

        for i_a in range(par.Na):
            ss.vbeg_l_a[i_fix, :, :, i_a] = ss.z_trans[i_fix] @ v_l_a[i_fix, :, :, i_a]



def evaluate_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. exogenous and targets
    ss.r = par.r_ss_target
    ss.rk = ss.r + par.delta_K
    ss.q = 1.0 / (1.0 + ss.r - par.delta_q)

    ss.N = 1.0  # normalization
    ss.Y = 1.0
    ss.K = par.K_Y_ratio*ss.Y
    ss.I = par.delta_K*ss.K
    ss.G = par.G_Y_ratio*ss.Y
    ss.qB = par.qB_Y_ratio*ss.Y
    ss.B = par.qB_Y_ratio*ss.Y/ss.q
    ss.L = par.L_Y_ratio*ss.Y
    # ss.hh_wealth = par.hh_wealth_Y_ratio*ss.Y
    # ss.A = ss.hh_wealth - ss.L
    ss.Q = 1.0

    ss.em = 0.0
    ss.i = ss.r

    # infered targets
    ss.s = (par.e_p - 1) / par.e_p
    ss.s_w = (par.e_w - 1) / par.e_w
    par.alpha = ss.rk * ss.K / ss.s / ss.Y
    ss.w = ss.s * (1 - par.alpha) * ss.Y / ss.N
    ss.q = 1 / (1 + ss.r - par.delta_q)
    ss.tau = (ss.G + ss.B + par.delta_q * ss.qB - ss.qB) \
             / ss.w
    ss.Z = (1 - ss.tau) * ss.w * ss.N
    ss.Z = (1 - ss.tau) * ss.w * ss.N

    par.Theta = ss.Y * ss.K ** (-par.alpha) * ss.N ** (par.alpha - 1)

    ss.ra = ss.r
    ss.rl = par.r_ss_target - par.xi

    ss.Div = ss.Y-ss.w*ss.N-ss.I
    ss.p_eq = ss.Div / ss.r
    ss.Div_k = ss.rk * ss.K - ss.I
    ss.p_k = ss.Div_k / ss.r
    ss.Div_int = 1-1/par.mu_p # alternatively ss.Div_int = ss.Div - ss.Div_k
    ss.p_int = ss.Div_int / ss.r

    ss.hh_wealth = ss.p_eq + ss.qB
    par.hh_wealth_Y_ratio = ss.hh_wealth # for easier access in backawrds solving
    ss.A = ss.hh_wealth - ss.L
    # ss.p_eq = ss.hh_wealth-ss.qB # old
    ss.p_share = ss.p_eq/ss.A
    # ss.ra2 = ss.p_share*(ss.Div+ss.p_eq)/ss.p_eq+(1-ss.p_share)*(1+ss.rl)-1

    ss.C = ss.Y - ss.G - ss.I - par.xi * ss.L
    # using the market clearing condision (see appendix C.6)
    ss.C = ss.w * ss.N + ss.Div - ss.G - par.xi * ss.L


    # if par.Nfix == 1:
    #     assert (1 + ss.r) * par.beta_mean < 1.0, '(1+r)*beta < 1, otherwise problems might arise'

    # b. relevant variables from household behavior
    model.solve_hh_ss(do_print=do_print)  # give us sol.a and sol.c
    if hasattr(par, 'start_dbeg_opti') and par.start_dbeg_opti: # start with optimized distribution along a grid
        ss.Dbeg = init_optimized_Dbeg(par, ss)
    model.simulate_hh_ss(do_print=do_print, Dbeg=ss.Dbeg)  # give us sim.D, ss.L_hh and ss.C_hh

    ss.Pi_w = 0.0
    v_prime_N_unscaled = ss.N ** (1 / par.frisch)
    u_prime_e = integrate_marg_util(ss.c, ss.D, par.z_grid, par.sigma)
    par.nu = ss.s_w * (1 - ss.tau) * ss.w * u_prime_e / v_prime_N_unscaled


def objective_ss(x, model, do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta_mean = x[0]

    evaluate_ss(model, do_print=do_print)

    market_clearing_LHS = ss.C_hh + ss.G + par.delta_K * ss.K + par.xi * ss.L_hh

    return ss.Y - market_clearing_LHS


def find_ss(model, do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    if do_print: print('Find optimal beta for market clearing')
    t0 = time.time()
    res = optimize.root(objective_ss, par.beta_mean, method='hybr', tol=par.tol_ss, args=(model))

    # final evaluation
    print('final evaluation')
    objective_ss(res.x, model, do_print=True)

    if par.Nfix == 1:
        assert (1 + ss.r) * res.x[0] < 1.0, '(1+r)*beta < 1, otherwise problems might arise'

    # check targets
    if do_print:
        print(f'steady state found in {elapsed(t0)}')
        print(f' beta   = {res.x[0]:8.8f}')
        # nu
        print(f'Implied nu = {par.nu:6.3f}')
        # market clearing
        market_clearing_LHS = ss.C_hh + ss.G + par.delta_K * ss.K + par.xi * ss.L_hh
        print(f'Discrepancy in C = {ss.C - ss.C_hh:12.8f} -> {(ss.C - ss.C_hh)/ss.C*100:12.6f}%')
        print(f'Discrepancy in L = {ss.L - ss.L_hh:12.8f} -> {(ss.L - ss.L_hh)/ss.L*100:12.6f}%')
        print(f'Discrepancy in A = {ss.A - ss.A_hh:12.8f} -> {(ss.A - ss.A_hh)/ss.A*100:12.6f}%')
        print(f'Market clearing residual = {ss.Y - market_clearing_LHS:12.8f}')
        # assert np.isclose(ss.Y, market_clearing_LHS)


def init_optimized_Dbeg(par, ss):
    """ initiate Dbeg that is optimized along the illiquid asset grid """

    # get distribution along z grid
    sigma = np.sqrt(par.sigma_e ** 2 * (1 - par.rho_e ** 2))
    _, _, e_ergodic, _, _ = log_rouwenhorst(par.rho_e, sigma, n=par.Ne)

    # find closest but smaller grid value to target
    i_a_target = np.abs(par.a_grid - ss.A).argmin()  # find grid value which is closest to the target
    if par.a_grid[i_a_target] > ss.A:
        i_a_target += -1  # select grid value that is smaller than target
    assert i_a_target <= par.Na, 'illiquid asset target outside of grid'
    # find weights between grid value and target,
    # s.t. w*a_grid[i]+(1-w)*a_grid[i+1] = a_target
    i_a_weight = (ss.A - par.a_grid[i_a_target + 1]) / (par.a_grid[i_a_target] - par.a_grid[i_a_target + 1])

    # fill Dbeg
    for i_fix in range(par.Nfix):
        ss.Dz[i_fix, :] = e_ergodic / par.Nfix
        for i_z in range(par.Nz):
            for i_l in range(par.Nl):
                ss.Dbeg[i_fix, i_z, i_l, :] = 0.0
                # distribute population shares along the relevant grids for the illiquid asset target
                ss.Dbeg[i_fix, i_z, i_l, i_a_target] = ss.Dz[i_fix, i_z] / par.Nl * i_a_weight
                ss.Dbeg[i_fix, i_z, i_l, i_a_target + 1] = ss.Dz[i_fix, i_z] / par.Nl * (1 - i_a_weight)
    return ss.Dbeg
