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

    # b. a, l
    par.a_grid[:] = equilogspace(par.a_min, par.a_max, par.Na)
    par.l_grid[:] = equilogspace(par.l_min, par.l_max, par.Nl)

    # c. e
    sigma = np.sqrt(par.sigma_e ** 2 * (1 - par.rho_e ** 2))
    par.z_grid[:], ss.z_trans[0, :, :], e_ergodic, _, _ = log_rouwenhorst(par.rho_e, sigma, n=par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    # start with single point distribution for each z
    # start with optimized distribution along a grid (saves 15 sec on my pc)
    if par.start_dbeg_opti:
        ss.Dbeg, ss.Dz = init_optimized_Dbeg(model, e_ergodic)
    else:
        for i_fix in range(par.Nfix):
            ss.Dz[i_fix, :] = e_ergodic / par.Nfix
            for i_z in range(par.Nz):
                ss.Dbeg[i_fix, i_z, :, :] *= 0.0
                ss.Dbeg[i_fix, i_z, 0, 0] = ss.Dz[i_fix, i_z]

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_l_a = np.zeros((par.Nfix, par.Nz, par.Nl, par.Na))
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):
            e = par.z_grid[i_z]
            Z = (1 - ss.tau) * ss.w * ss.N
            Ze = Z * e
            for i_a in range(par.Na):
                a_lag = par.a_grid[i_a]
                d = ss.ra * a_lag
                m = (1 + ss.rl) * par.l_grid + Ze + d
                c = m
                v_l_a[i_fix, i_z, :, i_a] = (1 + ss.rl) * c ** (-par.sigma)

        for i_a in range(par.Na):
            ss.vbeg_l_a[i_fix, :, :, i_a] = ss.z_trans[i_fix] @ v_l_a[i_fix, :, :, i_a]

    ###############
    # 4. warnings #
    ###############

    if par.print_non_lin_warning:
        lowest_income = (1 + ss.rl) * par.l_grid[0] + ss.Z * par.z_grid[0]
        highest_redistribution = ss.ra * par.a_grid[0] + par.chi * (
                    (1 + ss.ra) * par.a_grid[0] - (1 + ss.ra) * par.A_target)
        if lowest_income + highest_redistribution < 0:
            print("negative cash-on-hand possible given a/l grids and steady state values "
                  "-> non-linearities in policy functions")
            highest_chi = - (lowest_income - ss.ra * par.a_grid[0]) / (
                    (1 + ss.ra) * par.a_grid[0] - (1 + ss.ra) * par.A_target)
            print(f"highest chi possible for no non-linearities in the steady state is {highest_chi}")
        par.print_non_lin_warning = False


def evaluate_ss(model, do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. exogenous and targets
    ss.Y = 1.0  # normalization
    ss.N = 1.0  # normalization

    ss.A = par.hh_wealth_Y_ratio * ss.Y * par.A_L_ratio
    ss.L = par.hh_wealth_Y_ratio * ss.Y * (1.0 - par.A_L_ratio)

    ss.r = par.r_ss_target

    ss.K = par.K_Y_ratio * ss.Y
    ss.G = par.G_Y_ratio * ss.Y
    ss.qB = par.qB_Y_ratio * ss.Y
    # assert np.isclose(ss.A / ss.L, par.A_Y_ratio / par.L_Y_ratio)

    # par.mu_p = ss.Y / (ss.Y - ss.r * (ss.hh_wealth * ss.Y - ss.qB - ss.K))
    par.mu_p = ss.Y / (ss.Y - ss.r * (par.hh_wealth_Y_ratio * ss.Y - ss.qB - ss.K))
    par.e_p = par.mu_p/(par.mu_p-1)
    par.e_w = par.e_p

    # zero inflation
    ss.Pi = 0.0
    ss.Pi_w = 0.0
    ss.Pi_w_increase = 0.0
    ss.Pi_increase = 0.0

    # shocks
    ss.eg = 0.0
    ss.em = 0.0
    ss.ez = 0.0

    ss.eg_direct = 0.0
    ss.eg_distribution = 0.0
    ss.eg_debt = 0.0
    ss.eg_transfer = 0.0

    # b. central bank
    ss.i = ss.r

    # c. mutal fund
    ss.ra = ss.r
    ss.rl = ss.r - par.xi
    ss.q = 1.0 / (1.0 + ss.r - par.delta_q)

    ss.B = par.qB_Y_ratio * ss.Y / ss.q

    # d. intermediate goods
    ss.rk = ss.r + par.delta_K
    ss.s = (par.e_p - 1) / par.e_p
    assert np.isclose(ss.s, 1 / par.mu_p)
    par.alpha = ss.rk * ss.K / ss.s
    par.Theta = ss.Y * ss.K ** (-par.alpha) * ss.N ** (par.alpha - 1)
    assert np.isclose(par.Theta * ss.K ** par.alpha * ss.N ** (1 - par.alpha), ss.Y)
    ss.w = ss.s * (1 - par.alpha) / ss.N
    ss.wN = ss.w*ss.N
    ss.Div_int = (1 - ss.s) * ss.Y
    assert np.isclose(ss.Div_int, ss.Y - ss.w * ss.N - ss.rk * ss.K)
    ss.p_int = ss.Div_int / ss.r

    # e. capital firms
    ss.Q = 1.0
    ss.psi = 0.0
    ss.I = par.delta_K * ss.K
    ss.Ip = ss.I
    ss.Div_k = ss.rk * ss.K - ss.I
    ss.p_k = ss.Div_k / ss.r

    # f. all firms
    ss.Div = ss.Y - ss.w * ss.N - ss.I
    assert np.isclose(ss.Div - ss.Div_int - ss.Div_k, 0.0)
    ss.p_eq = ss.Div / ss.r
    assert np.isclose(ss.A, ss.p_eq + ss.qB - ss.L)

    # g. unions
    ss.s_w = (par.e_w - 1) / par.e_w

    # h. government
    ss.tau = (ss.G + (1 + par.delta_q * ss.q) * ss.B - ss.q * ss.B) / (ss.w * ss.N)

    # i. households
    ss.Z = (1 - ss.tau) * ss.w * ss.N
    par.A_target = ss.A
    assert par.Nfix == 1, NotImplementedError

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print, Dbeg=ss.Dbeg)


    v_prime_N_unscaled = ss.N ** (1 / par.frisch)
    u_prime_e = ss.UCE_hh
    par.nu = ss.s_w * (1 - ss.tau) * ss.w * u_prime_e / v_prime_N_unscaled

    # j. clearing
    ss.clearing_Y = ss.Y - (ss.C_hh + ss.G + ss.I + ss.psi + par.xi * ss.L_hh)
    ss.clearing_A = ss.A_hh - ss.A
    ss.clearing_L = ss.L_hh - ss.L
    ss.clearing_wealth = ss.A + ss.L - (ss.L_hh + ss.A_hh)



def objective_ss(x, model, do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta_mean = x[0]
    par.sigma_e = x[1]
    # par.hh_wealth_Y_ratio = x[1]
    par.A_L_ratio = x[2]

    evaluate_ss(model, do_print=do_print)

    model._compute_jac_hh()

    # MPC_annual = model.jac_hh[('C_hh', 'ez')][0, 0:4].sum()
    MPC_annual = np.sum([model.jac_hh[('C_hh', 'eg_transfer')][i, 0] / (1 + ss.r) ** i for i in range(4)])

    ss.MPC_target = par.MPC_target - MPC_annual

    return ss.MPC_target, ss.clearing_Y, ss.clearing_wealth


def find_ss(model, do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    if do_print: print('Find optimal beta for market clearing')

    t0 = time.time()
    res = optimize.root(objective_ss, np.array([par.beta_mean, par.sigma_e, par.A_L_ratio]), method='hybr', tol=par.tol_ss, args=(model))   # method: hybr

    assert res["success"], res["message"]

    # b. final evaluation
    if do_print: print('final evaluation')
    objective_ss([par.beta_mean, par.sigma_e, par.A_L_ratio], model, do_print=do_print)

    # check targets
    if do_print:
        print(f'steady state found in {elapsed(t0)}')
        print(f' beta   = {par.beta_mean:6.4}')
        print(f' nu     = {par.nu:6.4f}')

        # print(f'Discrepancy in C = {ss.clearing_C:12.8f}')
        print(f'Discrepancy in L = {ss.clearing_L:12.8f}')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')
        print(f'Discrepancy in hh wealth = {ss.clearing_wealth:12.8f}')
        print(f'Discrepancy from annual MPC target of {par.MPC_target} = {ss.MPC_target:12.8f}')


def init_optimized_Dbeg(model, e_ergodic):
    """ initiate Dbeg that is optimized along the illiquid asset grid """

    par = model.par
    ss = model.ss

    A_target = ss.A

    # find closest but smaller grid value to target
    i_a_target = np.abs(par.a_grid - A_target).argmin()  # find grid value which is closest to the target
    if par.a_grid[i_a_target] > A_target:
        i_a_target += -1  # select grid value that is smaller than target
    assert i_a_target <= par.Na, 'illiquid asset target outside of grid'
    # find weights between grid value and target,
    # s.t. w*a_grid[i]+(1-w)*a_grid[i+1] = a_target
    i_a_weight = (A_target - par.a_grid[i_a_target + 1]) / (par.a_grid[i_a_target] - par.a_grid[i_a_target + 1])

    # fill Dbeg
    Dbeg = np.zeros_like(ss.Dbeg)
    Dz =  np.zeros_like(ss.Dz)
    for i_fix in range(par.Nfix):
        Dz[i_fix, :] = e_ergodic / par.Nfix
        for i_z in range(par.Nz):
            for i_l in range(par.Nl):
                # distribute population shares along the relevant grids for the illiquid asset target
                Dbeg[i_fix, i_z, i_l, i_a_target] = Dz[i_fix, i_z] / par.Nl * i_a_weight
                Dbeg[i_fix, i_z, i_l, i_a_target + 1] = Dz[i_fix, i_z] / par.Nl * (1 - i_a_weight)
    # assert
    Dbeg_sum = np.sum(Dbeg)
    assert np.isclose(Dbeg_sum, 1.0), f'sum(ss.Dbeg) = {Dbeg_sum:12.8f}, should be 1.0'

    return Dbeg, Dz

