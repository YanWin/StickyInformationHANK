import numpy as np
import numba as nb

from GEModelTools import lag, lead, bound, bisection
# from GEModelTools.path import bisection_no_jit
from helper_functions import integrate_marg_util
from helper_functions import broyden_solver_cust
from helper_functions import residual_with_linear_continuation
from helper_functions import obtain_J



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
        Pi_increase = path.Pi_increase[ncol, :]
        Pi_w_increase = path.Pi_w_increase[ncol, :]
        G = path.G[ncol,:]
        tau = path.tau[ncol,:]
        B = path.B[ncol, :]
        Y = path.Y[ncol, :]
        N = path.N[ncol, :]
        I = path.I[ncol, :]
        K = path.K[ncol, :]
        Ip = path.Ip[ncol, :]
        # K_plus = path.K_plus[ncol, :]
        # K_plus2 = path.K_plus2[ncol, :]
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
        invest_res = path.invest_res[ncol, :]
        valuation_res = path.valuation_res[ncol, :]
        NKPC_res = path.NKPC_res[ncol, :]
        NKPC_w_res = path.NKPC_w_res[ncol, :]
        N_res = path.N_res[ncol, :]
        s_res = path.s_res[ncol, :]
        rk_res = path.rk_res[ncol, :]
        # k_res = path.k_res[ncol, :]
        # k_dummy = path.k_dummy[ncol, :]
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

        # Ip -> I and K
        for t in range(par.T):
            K_lag = K[t - 1] if t > 0 else ini.K
            if t == 0:
                I[t] = ini.I
            else:
                I[t] = Ip[t - 1]
            K[t] = (1 - par.delta_K) * K_lag + I[t]

        N_res[:] = N -  (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
        s_res[:] = s - w * N / Y / (1 - par.alpha)
        rk_res[:] = rk - s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)
        # N[:] = (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
        # s[:] = w * N / Y / (1 - par.alpha)
        # rk[:] = s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)

        # Q
        for t_ in range(par.T):
            t = par.T - 1 - t_
            Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
            r_plus = r[t + 1] if t < par.T - 1 else ss.r
            rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk

            Q_t = 1.0 / (1.0 + r_plus) * (rk_plus2 + (1.0 - par.delta_K) * Q_plus)
            valuation_res[t] = Q_t - Q[t]

            # investment residual
        for t_ in range(par.T):  # par.T-1
            t = par.T - 1 - t_
            # if t < par.T - 2:
            Ip_plus = Ip[t + 1] if t < par.T - 1 else ss.I
            r_plus = r[t + 1] if t < par.T - 1 else ss.r

            S = par.phi_K / 2 * (Ip[t] / I[t] - 1.0) ** 2
            Sderiv = par.phi_K * (Ip[t] / I[t] - 1.0)
            Sderiv_plus = par.phi_K * (Ip_plus / Ip[t] - 1.0)

            LHS = 1.0 + S + Ip[t] / I[t] * Sderiv
            RHS = Q[t] + 1.0 / (1.0 + r_plus) * (Ip_plus / Ip[t]) ** 2 * Sderiv_plus
            invest_res[t] = RHS - LHS
            # elif t == par.T - 2:
            #     invest_res[t] = Ip[t] - (ss.K - (1 - par.delta_K) * K[t + 1])
            # else:
            #     invest_res[t] = Ip[t] - ss.I

        ###
        # b. NKPC prices block
        ###
            # input: s
            # output: Pi

        # # NKPC in the form Pi[t] - Pi[t - 1] = kappa * E(sum(s-ss.s))
        # for t_ in range(par.T):
        #     t = (par.T - 1) - t_
        #     kappa = (1 - par.xi_p) * (1 - par.xi_p / (1 + ss.r)) / par.xi_p \
        #             * par.e_p / (par.v_p + par.e_p - 1)
        #     gap = 0
        #     for k in range(t_+1):
        #         gap += (1 / (1 + ss.r)) ** k * (s[t + k] - (par.e_p - 1) / par.e_p)
        #     Pi_increase[t] = kappa * gap

        # NKPC in the form Pi[t] - Pi[t - 1] = a + b *  E(Pi[t + 1] - Pi[t])
        for t_ in range(par.T):
            t = (par.T - 1) - t_
            Pi_increase_plus = Pi_increase[t + 1] if t < par.T - 1 else 0
            kappa = (1 - par.xi_p) * (1 - par.xi_p / (1 + r[t])) / par.xi_p \
                    * par.e_p / (par.v_p + par.e_p - 1)
            Pi_increase[t] = kappa * (s[t] - (par.e_p - 1) / par.e_p) + (1 / (1 + r[t])) * Pi_increase_plus

        for t in range(par.T):
            Pi_lag = Pi[t - 1] if t > 0 else ini.Pi
            Pi_t = Pi[t - 1] + Pi_increase[t]
            NKPC_res[t] = Pi_t - Pi[t]


        ###
        # c. Taylor rule block
        ###
            # inputs: Pi
            # outputs: i
        for t in range(par.T):
            i_lag = i[t - 1] if t > 0 else ini.i
            i[t] = (1 + ss.r) ** (1 - par.rho_m) * (1 + i_lag) ** (par.rho_m) \
                   * (1 + Pi[t]) ** ((1 - par.rho_m) * par.phi_pi) * (1 + em[t]) - 1
            # simple taylor rule
            # i[t] = par.rho_m * i_lag + (1 - par.rho_m) * (ss.r + par.phi_pi * Pi[t]) + em[t]

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
        for t in range(par.T):
            I_lag = I[t - 1] if t > 0 else ini.I
            S = par.phi_K / 2 * (I[t] / I_lag - 1.0) ** 2
            psi[t] = I[t] * S
        for t in range(par.T):
            # Div
            Div[t] = Y[t] - w[t] * N[t] - I[t] - psi[t]
            # Div_k
            Div_k[t] = rk[t] * K[t] - I[t] - psi[t]
            # Div_int
            Div_int[t] = Div[t] - Div_k[t]

        for t_ in range(par.T):
            t = (par.T - 1) - t_
            # q
            q_plus = q[t + 1] if t < par.T - 1 else ss.q
            q[t] = (1 + par.delta_q * q_plus) / (1 + r[t])
            # p_eq
            p_eq_plus = p_eq[t + 1] if t < par.T - 1 else ss.p_eq
            Div_plus = Div[t + 1] if t < par.T - 1 else ss.Div
            p_eq[t] = (Div_plus + p_eq_plus) / (1 + r[t])
            # p_k
            Div_k_plus = Div_k[t + 1] if t < par.T - 1 else ss.Div_k
            p_k_plus = p_k[t + 1] if t < par.T - 1 else ss.p_k
            p_k[t] = (1 / (1 + r[t])) * (p_k_plus + Div_k_plus)
            # p_int
            Div_int_plus = Div_int[t + 1] if t < par.T - 1 else ss.Div_int
            p_int_plus = p_int[t + 1] if t < par.T - 1 else ss.p_int
            p_int[t] = (1 / (1 + r[t])) * (p_int_plus + Div_int_plus)

        # ra
        # dA = lambda A, ra : par.chi * ((1 + ra) * A - (1 + ss.r) * par.A_target)
        dA = lambda A, ra_t: ss.ra / (1 + ss.ra) * (1 + ra_t) * A + par.chi * ((1 + ra_t) * A - (1 + ss.ra) * par.A_target)
        A_t = par.A_target
        for t in range(par.T):
            p_eq_lag = p_eq[t - 1] if t > 0 else ini.p_eq
            q_lag = q[t - 1] if t > 0 else ini.q
            p_share_lag = p_share[t - 1] if t > 0 else ini.p_share
            A_lag = A_t
            ra[t] = p_share_lag * (Div[t] + p_eq[t]) / p_eq_lag + \
                    (1 - p_share_lag) * (1 + par.delta_q * q[t]) / q_lag -1
            A_t = (1 + ra[t]) * A_lag - dA(A_lag, ra[t])
            p_share[t] = p_eq[t] / A_t


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
            Z[t] = (1 - tau[t]) * w[t] * N[t]

        # without fiscal schock:
        # G[:] = ss.G
        # for t in range(par.T):
        #     B_lag = B[t-1] if t > 0 else ini.B
        #     tau[t] = par.phi_tau * ss.q * (B_lag - ss.B) / ss.Y + ss.tau
        #     B[t] = (G[t] + (1 + par.delta_q * q[t]) * B_lag - tau[t] * w[t] * N[t]) / q[t]
        #     Z[:] = (1 - tau[t]) * w[t] * N[t]
        #     qB[t] = q[t] * B[t]

        # to test errors in the model
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
        Pi_increase = path.Pi_increase[ncol, :]
        Pi_w_increase = path.Pi_w_increase[ncol, :]
        G = path.G[ncol,:]
        tau = path.tau[ncol,:]
        B = path.B[ncol, :]
        Y = path.Y[ncol, :]
        N = path.N[ncol, :]
        I = path.I[ncol, :]
        K = path.K[ncol, :]
        Ip = path.Ip[ncol, :]
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
        invest_res = path.invest_res[ncol, :]
        valuation_res = path.valuation_res[ncol, :]
        NKPC_res = path.NKPC_res[ncol, :]
        NKPC_w_res = path.NKPC_w_res[ncol, :]
        N_res = path.N_res[ncol, :]
        s_res = path.s_res[ncol, :]
        rk_res = path.rk_res[ncol, :]
        # k_res = path.k_res[ncol, :]
        # k_dummy = path.k_dummy[ncol, :]
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
            da = ss.ra / (1 + ss.ra) * (1 + ra[t]) * a[t] + par.chi * ((1 + ra[t]) * a[t] - (1 + ss.ra) * A_target)
            d_agg = np.sum(da * D[t])
            A[t] = (1 + ra[t]) * A_lag - d_agg

        L[:] = hh_wealth - A

        for t in range(par.T):
            L_lag = L[t - 1] if t > 0 else ini.L
            C[t] = Y[t] - G[t] - I[t] - psi[t]- par.xi * L_lag

        ###
        # a. NKPC-wage block
        ###
            # inputs C_hh, w, tau, pi
            # outputs: Pi_w

        # # NKPC-wage given in the form Pi_w_t - Pi_t = kappa_w * E[sum(s_w - ss.s_w)]
        # kappa_w = (1 - par.xi_w) * (1 - par.xi_w * par.beta_mean) / par.xi_w \
        #           * par.e_w / (par.v_w + par.e_w - 1)
        # for t_ in range(par.T):
        #     t = (par.T - 1) - t_
        #     s_w[t] = par.nu * N[t] ** (1 / par.frisch) / ((1 - tau[t]) * w[t] * UCE_hh[t])
        #     gap_w = 0
        #     for k in range(t_+1):
        #         gap_w += par.beta_mean ** k * (s_w[t + k] - (par.e_w - 1) / par.e_w)
        #     Pi_w_increase[t] = kappa_w * gap_w

        # NKPC-wage given in the form Pi_w_t - Pi_t = a + b *  E(Pi_w[t + 1] - Pi[t])
        kappa_w = (1 - par.xi_w) * (1 - par.xi_w * par.beta_mean) / par.xi_w \
                  * par.e_w / (par.v_w + par.e_w - 1)
        for t_ in range(par.T):
            t = (par.T - 1) - t_
            Pi_w_increase_plus = Pi_w_increase[t + 1] if t < par.T - 1 else ss.Pi_w - ss.Pi
            s_w[t] = par.nu * N[t] ** (1 / par.frisch) / ((1 - tau[t]) * w[t] * UCE_hh[t])
            Pi_w_increase[t] = kappa_w * (s_w[t] - (par.e_w - 1)/ par.e_w) + par.beta_mean * Pi_w_increase_plus

        for t in range(par.T):
            Pi_lag = Pi[t - 1] if t > 0 else ss.Pi
            Pi_w_t = Pi_lag + Pi_w_increase[t]
            NKPC_w_res[t] = Pi_w_t - Pi_w[t]

        ###########
        # targets #
        ###########

        # Fisher equation (with r as ex-ante real interest rate)
        for t_ in range(par.T):
            t = (par.T - 1) - t_
            Pi_plus = Pi[t + 1] if t < par.T - 1 else ss.Pi
            fisher_res[t] = 1 + i[t] - (1 + r[t]) * (1 + Pi_plus)

        # wage residual
        for t in range(par.T):
            w_lag = w[t - 1] if t > 0 else ini.w
            w_res[t] = np.log(w[t] / w_lag) - (Pi_w[t] - Pi[t])

        # Good market clearing
        for t in range(par.T):
            L_hh_lag = L_hh[t - 1] if t > 0 else ini.L_hh
            clearing_Y[t] = C_hh[t] + I[t] + G[t] + psi[t] + par.xi * L_hh_lag - Y[t]

