import numpy as np
import numba as nb

from GEModelTools import lag, lead, bound, bisection
# from GEModelTools.path import bisection_no_jit
from helper_functions import integrate_marg_util
from helper_functions import broyden_solver_cust
from helper_functions import residual_with_linear_continuation
from helper_functions import obtain_J


@nb.njit
def NKPC_eq(x, par, r, s, Pi_plus):
    gap = s - (par.e_p - 1)/par.e_p

    kappa = (1-par.xi_p) * (1-par.xi_p/(1+r)) / par.xi_p \
            * par.e_p / (par.v_p + par.e_p - 1)

    NKPC = x - 1/(1+r) * kappa * gap - 1/(1+r)*Pi_plus

    return NKPC

@nb.njit
def NKPC_w_eq(x, par, s_w, Pi_w_plus):
    gap = s_w - (par.e_w - 1)/par.e_w

    kappa = (1-par.xi_w) * (1-par.xi_w*par.beta_mean) / par.xi_w \
            * par.e_w / (par.v_w + par.e_w - 1)

    NKPC_w = x - kappa * gap - par.beta_mean*Pi_w_plus

    return NKPC_w


# @nb.njit
# def unpack_kwargs(d):
#     """ simple unpacking funtion specific to the current residual function"""
#     return d['par'], d['ss'], d['Y'], d['w'], d['r'], d['t_predet']
# 
# @nb.njit
# def flat_to_I_Q(x, t_predet, T, ss):
#     """ Flat array into seperate arrays for I_t+1/I_t and Q"""
#     I = np.empty(T)
#     Q = np.empty(T)
# 
#     # mask_I = mask_Q = np.ones(T, bool)
#     # mask_I[t_predet['I_plus_div_I']] = False
#     # mask_Q[t_predet['Q']] = False
# 
#     # for numba
#     mask_I = np.ones(T,  nb.bool_)
#     mask_Q = np.ones(T,  nb.bool_)
#     for v in t_predet['I']:
#         mask_I[v] = False
#     for v in t_predet['Q']:
#         mask_Q[v] = False
# 
#     I[mask_I] = x[:(T - len(t_predet['I']))]
#     I[np.invert(mask_I)] = ss.I
# 
#     Q[mask_Q] = x[(T - len(t_predet['I'])):]
#     Q[np.invert(mask_Q)] = ss.Q
# 
#     return I, Q


@nb.njit
def unpack_kwargs(d):
    """ simple unpacking funtion specific to the current residual function"""
    return d['par'], d['ss'], d['Y'], d['w'], d['r']

@nb.njit
def adj_costs(I_frac, phi, delta_K):
    adj_costs = phi / 2 * (I_frac - 1.0) ** 2
    adj_costs_deriv1 = phi * (I_frac - 1.0)
    adj_costs_deriv2 = phi
    return adj_costs, adj_costs_deriv1, adj_costs_deriv2

@nb.njit
def inv_eq(Q, I_frac, I_frac_plus, r_plus, delta_K, phi_K):
    S, S1, _ = adj_costs(I_frac, phi_K, delta_K)
    _, S1_plus, _ = adj_costs(I_frac_plus, phi_K, delta_K)
    LHS = 1.0 + S + I_frac * S1
    RHS = Q + (1.0 / (1.0 + r_plus)) * I_frac_plus ** 2 * S1_plus
    inv_target = LHS - RHS
    # for numerical stability. Otherwise dx in solving the jacobian gets out of hand
    if abs(inv_target) < 1e-12:
        inv_target = 0
    return inv_target

@nb.njit
def residual(x, kwargs_dict):
    """ residual function to optimize using the broyden solver

        :arg x: flattened np.array containing the unknowns"""

    # unpack
    par, ss, Y, w, r = unpack_kwargs(kwargs_dict)

    # get values for K and Q. For predetermined values take ss values
    I_frac = x[:par.T]
    Q = x[par.T:]

    # init arrays
    K = np.empty(par.T + 2)
    I = np.empty(par.T + 1)
        # targets
    target1 = np.empty(par.T)
    target2 = np.empty(par.T)
        # labor block
    N = np.empty_like(K)
    s = np.empty_like(K)
    rk = np.empty_like(K)

    for t in range(par.T):
        if t == 0:
            I[t] = ss.I
            K[t] = ss.K
            I[t + 1] = I_frac[t] * I[t]
            K[t + 1] = (1 - par.delta_K) * K[t] + I[t]
        elif t < par.T - 1:
            I[t + 1] = I_frac[t] * I[t]
            K[t + 1] = (1 - par.delta_K) * K[t] + I[t]
        else:
            # needed to calculate rk_t+2
            K[par.T] = (1 - par.delta_K) * K[par.T - 1] + I[par.T - 1]
            I[par.T] = I_frac[par.T - 1] * I[par.T - 1]
            K[par.T + 1] = (1 - par.delta_K) * K[par.T] + I[par.T]

    # predetermined values
    assert abs(I[0] - ss.I) < 1e-12, 'I[0] != ss.I'

    # calculate labor block
    for t in range(par.T + 2):
        if t < par.T:
            N[t] = (Y[t] / (par.Theta * K[t] ** par.alpha)) ** (1 / (1 - par.alpha))
            s[t] = w[t] * N[t] / Y[t] / (1 - par.alpha)
            rk[t] = s[t] * par.alpha * par.Theta * K[t] ** (par.alpha - 1) * N[t] ** (1 - par.alpha)
        else:
            N[t] = (ss.Y / (par.Theta * K[t] ** par.alpha)) ** (1 / (1 - par.alpha))
            s[t] = w[t] * N[t] / ss.Y / (1 - par.alpha)
            rk[t] = s[t] * par.alpha * par.Theta * K[t] ** (par.alpha - 1) * N[t] ** (1 - par.alpha)

    # calculate values for target equation
    for t in range(par.T):
        # Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
        # r_plus = r[t + 1] if t < par.T - 1 else ss.r
        # rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk
        # I_frac_plus =  I_frac[t + 1] if t < par.T - 1 else 1
        #
        # # calculate targets
        # Q_t = (1 / (1 + r_plus)) * (rk_plus2 + (1 - par.delta_K) * Q_plus)
        # target1[t] = inv_eq(Q[t], I_frac[t], I_frac_plus, r_plus, par.delta_K, par.phi_K)
        # target2[t] = Q[t] - Q_t

        Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
        r_plus = r[t + 1] if t < par.T - 1 else ss.r
        rk_plus2 = rk[t + 2]
        I_frac_plus =  I_frac[t + 1] if t < par.T - 1 else ss.I/I[par.T]

        # calculate targets
        Q_t = (1 / (1 + r_plus)) * (rk_plus2 + (1 - par.delta_K) * Q_plus)
        target1[t] = inv_eq(Q[t], I_frac[t], I_frac_plus, r_plus, par.delta_K, par.phi_K)
        target2[t] = Q[t] - Q_t

    return np.hstack((target1, target2))



@nb.njit
def block_pre(par, ini, ss, path, ncols=1):
    """ evaluate transition path - before household block """
    for ncol in nb.prange(ncols):
        r = path.r[ncol,:]
        ra = path.ra[ncol,:]
        rl = path.rl[ncol,:]
        i = path.i[ncol,:]
        Pi = path.Pi[ncol,:]
        Pi_w = path.Pi_w[ncol,:]
        G = path.G[ncol,:]
        tau = path.tau[ncol,:]
        B = path.B[ncol, :]
        Y = path.Y[ncol, :]
        N = path.N[ncol, :]
        I = path.I[ncol, :]
        K = path.K[ncol, :]
        Div = path.Div[ncol, :]
        Q = path.Q[ncol, :]
        C = path.C[ncol, :]
        L = path.L[ncol, :]
        A = path.A[ncol, :]
        C_hh = path.C_hh[ncol, :]
        L_hh = path.L_hh[ncol, :]
        A_hh = path.A_hh[ncol, :]
        UCE_hh = path.UCE_hh[ncol, :]
        # UCE_hh = path.UCE_hh[ncol, :]
        qB = path.qB[ncol, :]
        w = path.w[ncol, :]
        q = path.q[ncol, :]
        hh_wealth = path.hh_wealth[ncol, :]
        clearing_Y = path.clearing_Y[ncol, :]
        fisher_res = path.fisher_res[ncol, :]
        w_res = path.w_res[ncol, :]
        em = path.em[ncol, :]
        eg = path.eg[ncol, :]
        Z = path.Z[ncol, :]
        s = path.s[ncol, :]
        rk = path.rk[ncol, :]
        s_w = path.s_w[ncol, :]
        psi = path.psi[ncol, :]
        # nu = path.nu[ncol, :]
        # Theta = path.Theta[ncol, :]
        p_eq = path.p_eq[ncol, :]
        p_share = path.p_share[ncol, :]
        p_k = path.p_k[ncol, :]
        Div_k = path.Div_k[ncol, :]
        p_int = path.p_int[ncol, :]
        Div_int = path.Div_int[ncol, :]
        c = path.c
        a = path.a
        D = path.D

        #################
        # implied paths #
        #################

        ###
        # a. Production Block
        ###
            # inputs: r,w,Y
            # outputs: D,N,I,s

        init_I_frac = np.ones_like(Y)
        init_Q = np.ones_like(Y)

        f_args = {'par': par,
                  'ss': ss,
                  'Y': Y,
                  'w': w,
                  'r': r}

        x0 = np.hstack((init_I_frac, init_Q))
        y0 = residual(x0, kwargs_dict=f_args)
        jac = obtain_J(residual, x0, y0, kwargs_dict=f_args)
        # with jac=None: calculate jacobian again each iteration
        x_end = broyden_solver_cust(residual, x0, kwargs_dict=f_args, use_jac=None,
                                    tol=1e-8, max_iter=200, backtrack_fac=0.5, max_backtrack=100,
                                    do_print=False)
        # back out I, K, Q
        I_frac_opt = x_end[:par.T]
        Q_opt = x_end[par.T:]
        Q[:] = Q_opt

        for t in range(par.T):
            if t == 0:
                I[t] = ss.I
                K[t] = ss.K
                I[t + 1] = I_frac_opt[t] * I[t]
                K[t + 1] = (1 - par.delta_K) * K[t] + I[t]
            elif t < par.T - 1:
                I[t + 1] = I_frac_opt[t] * I[t]
                K[t + 1] = (1 - par.delta_K) * K[t] + I[t]
            else:
                pass

        assert abs(K[0] - ss.K) < 1e-10, 'K[0] != ss.K'
        assert abs(K[1] - ss.K) < 1e-10, 'K[1] != ss.K'
        # assert abs(
        #     (1 - par.delta_K) * K[par.T - 1] + I[par.T - 1] - ss.K) < 1e-7, 'K_T != ss.K'


        N[:] = (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
        s[:] = w * N / Y / (1 - par.alpha)
        rk[:] = s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)



        ###
        # b. NKPC prices block
        ###
            # input: s
            # output: Pi
        for t_ in range(par.T):
            t = (par.T - 1) - t_
            Pi_plus = Pi[t + 1] if t < par.T - 1 else ss.Pi
            # Pi[t] = bisection_no_jit(NKPC_eq, -0.2, 0.2, args=(par, r[t], s[t], Pi_plus))
            Pi[t] = bisection(NKPC_eq, -0.2, 0.2, args=(par, r[t], s[t], Pi_plus))

        ###
        # c. Taylor rule block
        ###
            # inputs: Pi
            # outputs: i
        for t in range(par.T):
            i_lag = i[t - 1] if t > 0 else ini.i
            # for monetary policy shock use
            # i[t] = (1 + ss.r) ** (1 - par.rho_m) * (1 + i_lag) ** (par.rho_m) \
            #        * (1 + Pi[t]) ** ((1 - par.rho_m) * par.phi_pi) * (1 + em[t]) - 1
            # simple taylor rule
            i[t] = par.rho_m * i_lag + (1 - par.rho_m) * (ss.r + par.phi_pi * Pi[t]) + em[t]

        ###
        # d. Finance block
        ###
            # Inputs: Div, r
            # outputs: q, rl, ra

        for t in range(par.T):
            # rl
            r_lag = r[t - 1] if t > 0 else ini.r
            rl[t] =  r_lag - par.xi

        # Dividends
        for t_ in range(par.T):
            t = (par.T - 1) - t_

            # Div
            # K_plus = K[t + 1] if t < par.T - 1 else ss.K
            # S, _, _ = adj_costs(K[t], K_plus, par.phi_K, par.delta_K)
            I_plus = I[t + 1] if t < par.T - 1 else ss.I
            S, _, _ = adj_costs(I_plus/I[t], par.phi_K, par.delta_K)
            psi[t] = I[t] * S
            Div[t] = Y[t] - w[t] * N[t] - I[t] - psi[t]

            # q
            q_plus = q[t + 1] if t < par.T - 1 else ss.q
            q[t] = (1 + par.delta_q * q_plus) / (1 + r[t])

            # p
            p_eq_plus = p_eq[t + 1] if t < par.T - 1 else ss.p_eq
            Div_plus = Div[t + 1] if t < par.T - 1 else ss.Div
            p_eq[t] = (Div_plus + p_eq_plus) / (1 + r[t])

            # Div_k
            Div_k[t] = rk[t] * K[t] - I[t] - psi[t]
            # Div_int
            Div_int[t] = Div[t] - Div_k[t]

            Div_k_plus = Div_k[t + 1] if t < par.T - 1 else ss.Div_k
            p_k_plus = p_k[t + 1] if t < par.T - 1 else ss.p_k
            p_k[t] = (1 / (1 + r[t])) * (p_k_plus + Div_k_plus)

            Div_int_plus = Div_int[t + 1] if t < par.T - 1 else ss.Div_int
            p_int_plus = p_int[t + 1] if t < par.T - 1 else ss.p_int
            p_int[t] = (1 / (1 + r[t])) * (p_int_plus + Div_int_plus)


        # for t in range(par.T):
        #     # ra
        #     p_eq_lag = p_eq[t - 1] if t > 0 else ini.p_eq
        #     q_lag = q[t - 1] if t > 0 else ini.q
        #     p_share[t] = p_eq_lag / (par.hh_wealth_Y_ratio - par.L_Y_ratio)
        #     ra[t] = p_share[t] * (Div[t] + p_eq[t]) / p_eq_lag \
        #          + (1 - p_share[t]) * (1 + par.delta_q * q[t]) / q_lag - 1

        dA = lambda A, ra : par.chi * ((1 + ra) * A - (1 + ss.r) * par.A_target)

        for t in range(par.T):
            p_int_lag = p_int[t - 1] if t > 0 else ini.p_int
            q_lag = q[t - 1] if t > 0 else ini.q
            A_lag = A[t - 1] if t > 0 else par.A_target
            p_share_lag = p_share[t - 1] if t > 0 else ini.p_share

            ra[t] = p_share_lag * (Div_int[t] + p_int[t]) / p_int_lag + \
                    (1 - p_share_lag) * (1 + par.delta_q * q[t]) / q_lag -1
            A_t = A_lag - dA(A_lag, ra[t])
            p_share[t] = p_int[t] / A_t

        # # this is just so see what happens if there isnt this big movement in ra[0] for a mp shock
        # for t in range(par.T):
        #     A_lag = A[t - 1] if t > 0 else par.A_target
        #     A_t = A_lag - dA(A_lag, ra[t])
        #     p_share[t] = p_int[t] / A_t
        # for t_ in range(par.T):
        #     t = (par.T - 1) - t_
        #     if t == 0:
        #         ra[t] == ss.ra
        #     if t < par.T-1:
        #         ra[t+1] =  p_share[t] * (Div_int[t+1] + p_int[t+1]) / p_int[t] + \
        #             (1 - p_share[t]) * (1 + par.delta_q * q[t+1]) / q[t] - 1




        # for t in range(par.T):
        #     # ra
        #     p_eq_lag = p_eq[t - 1] if t > 0 else ini.p_eq
        #     q_lag = q[t - 1] if t > 0 else ini.q
        #     hh_wealth_lag = hh_wealth[t - 1] if t > 0 else ini.hh_wealth
        #     p_share[t] = p_eq / hh_wealth
        #     ra[t] = p_share[t] * (Div[t] + p_eq[t]) / p_eq_lag \
        #          + (1 - p_share[t]) * (1 + par.delta_q * q[t]) / q_lag - 1


        ###
        # e. Fiscal block
        ###
            # Inputs: q, w, eg
            # Outputs: tau, Z, G
        G[:] = ss.G * (1 + eg) # constant government spending

        for t in range(par.T):
            B_lag = B[t-1] if t > 0 else ini.B
            tau_no_shock =  par.phi_tau * ss.q * (B_lag - ss.B) / ss.Y + ss.tau
            B_no_shock = (ss.G + (1 + par.delta_q * q[t]) * B_lag - tau_no_shock * w[t] * N[t]) / q[t]
            delta_tau = ((1-par.phi_G) * ss.G * eg[t]) / w[t] / N[t]
            delta_B = par.phi_G * ss.G * eg[t] / q[t]
            tau[t] = delta_tau + tau_no_shock
            B[t] = delta_B + B_no_shock
            # value of government debt
            qB[t] = q[t] * B[t]
            # labor income without idiosyncratic shocks
            Z[:] = (1 - tau[t]) * w[t] * N[t]

        # without fiscal schock:
        # G[:] = ss.G
        # for t in range(par.T):
        #     B_lag = B[t-1] if t > 0 else ini.B
        #     tau[t] = par.phi_tau * ss.q * (B_lag - ss.B) / ss.Y + ss.tau
        #     B[t] = (G[t] + (1 + par.delta_q * q[t]) * B_lag - tau[t] * w[t] * N[t]) / q[t]
        #     Z[:] = (1 - tau[t]) * w[t] * N[t]
        #     qB[t] = q[t] * B[t]



        hh_wealth[:] = p_eq + qB






@nb.njit
def block_post(par,ini,ss,path,ncols=1):
    """ evaluate transition path - after household block """

    for ncol in nb.prange(ncols):
        r = path.r[ncol,:]
        ra = path.ra[ncol,:]
        rl = path.rl[ncol,:]
        i = path.i[ncol,:]
        Pi = path.Pi[ncol,:]
        Pi_w = path.Pi_w[ncol,:]
        G = path.G[ncol,:]
        tau = path.tau[ncol,:]
        B = path.B[ncol, :]
        Y = path.Y[ncol, :]
        N = path.N[ncol, :]
        I = path.I[ncol, :]
        K = path.K[ncol, :]
        Div = path.Div[ncol, :]
        Q = path.Q[ncol, :]
        C = path.C[ncol, :]
        L = path.L[ncol, :]
        A = path.A[ncol, :]
        C_hh = path.C_hh[ncol, :]
        L_hh = path.L_hh[ncol, :]
        A_hh = path.A_hh[ncol, :]
        UCE_hh = path.UCE_hh[ncol, :]
        qB = path.qB[ncol, :]
        w = path.w[ncol, :]
        q = path.q[ncol, :]
        hh_wealth = path.hh_wealth[ncol, :]
        clearing_Y = path.clearing_Y[ncol, :]
        fisher_res = path.fisher_res[ncol, :]
        w_res = path.w_res[ncol, :]
        em = path.em[ncol, :]
        eg = path.eg[ncol, :]
        Z = path.Z[ncol, :]
        s = path.s[ncol, :]
        rk = path.rk[ncol, :]
        s_w = path.s_w[ncol, :]
        psi = path.psi[ncol, :]
        # nu = path.nu[ncol, :]
        # Theta = path.Theta[ncol, :]
        p_eq = path.p_eq[ncol, :]
        p_share = path.p_share[ncol, :]
        p_k = path.p_k[ncol, :]
        Div_k = path.Div_k[ncol, :]
        p_int = path.p_int[ncol, :]
        Div_int = path.Div_int[ncol, :]
        c = path.c
        a = path.a
        D = path.D

        #################
        # implied paths #
        #################


        A_target = (par.hh_wealth_Y_ratio - par.L_Y_ratio)*ss.Y
        for t in range(par.T):
            A_lag = A[t - 1] if t > 0 else ini.A
            da = ss.r / (1 + ss.r) * (1 + ra[t]) * a[t] + par.chi * ((1 + ra[t]) * a[t] - (1 + ss.r) * A_target)
            d_agg = np.sum(da * D[t])
            A[t] = (1 + ra[t]) * A_lag - d_agg

        L[:] = hh_wealth - A


        for t in range(par.T):
            # A_lag = A[t - 1] if t > 0 else ini.A
            L_lag = L[t - 1] if t > 0 else ini.L
            # rl_lag = rl[t - 1] if t > 0 else ini.rl
            C[t] = Y[t] - G[t] - I[t] - psi[t] - par.xi * L_lag
            # C[t] = (1 + rl_lag) * L_lag + (1 + ra[t]) * A_lag + (1 - tau[t]) * w[t] * N[t] - A[t] - L[t]

        ###
        # a. NKPC-wage block
        ###
            # inputs C_hh, w, tau, pi
            # outputs: Pi_w
        for t_ in range(par.T):
            t = (par.T - 1) - t_
            s_w[t] = par.nu * N[t] ** (1 / par.frisch) / ((1 - tau[t]) * w[t] * UCE_hh[t])
            # u_prime_e = integrate_marg_util(c[t], D[t], par.z_grid, par.sigma)
            # s_w[t] = par.nu * N[t] ** (1/par.frisch) / ( (1 - tau[t]) * w[t] * u_prime_e)

            Pi_w_plus = Pi_w[t + 1] if t < par.T - 1 else ss.Pi_w
            Pi_w[t] = bisection(NKPC_w_eq, -0.2, 0.2, args=(par, s_w[t], Pi_w_plus))

        ###########
        # targets #
        ###########

        # Fisher equation (with r as ex-ante real interest rate)
        for t_ in range(par.T):
            t = (par.T - 1) - t_
            Pi_plus = Pi[t + 1] if t < par.T - 1 else ss.Pi
            fisher_res[t] = 1 + i[t] - (1 + r[t]) * (1 + Pi_plus)

        # # (ex-post) Fisher equation
        # for t in range(par.T):
        #     i_lag = i[t - 1] if t > 0 else ini.i
        #     fisher_res[t] = 1 + i_lag - (1 + r[t]) * (1 + Pi[t])

        # # Fisher residual from their paper (I think it is wrong)
        # for t_ in range(par.T):
        #     t = (par.T - 1) - t_
        #     Pi_plus = Pi[t + 1] if t < par.T - 1 else ss.Pi
        #     # r_plus = r[t + 1] if t < par.T - 1 else ss.r
        #     fisher_res[t] = 1 + r[t] - (1 + i[t]) * (1 + Pi_plus)


        # wage residual
        for t in range(par.T):
            w_lag = w[t - 1] if t > 0 else ini.w
            w_res[t] = np.log(w[t] / w_lag) - (Pi_w[t] - Pi[t])

        # Good market clearing
        for t in range(par.T):
            L_hh_lag = L_hh[t - 1] if t > 0 else ini.L_hh
            # C[t] = Y[t] - G[t] - I[t] - I[t] * 0 - par.xi * L_hh_lag
            clearing_Y[t] = C_hh[t] + I[t] + G[t] + psi[t] + par.xi * L_hh_lag - Y[t]










