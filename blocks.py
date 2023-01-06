import numpy as np
import numba as nb

from GEModelTools import lag, lead


@nb.njit
def block_pre(par, ini, ss, path, ncols=1):
    """ evaluate transition path - before household block """

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol, :]
        B = path.B[ncol, :]
        clearing_A = path.clearing_A[ncol, :]
        clearing_L = path.clearing_L[ncol, :]
        clearing_Y = path.clearing_Y[ncol, :]
        Div_int = path.Div_int[ncol, :]
        Div_k = path.Div_k[ncol, :]
        Div = path.Div[ncol, :]
        eg = path.eg[ncol, :]
        em = path.em[ncol, :]
        ez = path.ez[ncol, :]
        fisher_res = path.fisher_res[ncol, :]
        G = path.G[ncol, :]
        i = path.i[ncol, :]
        I = path.I[ncol, :]
        invest_res = path.invest_res[ncol, :]
        Ip = path.Ip[ncol, :]
        K = path.K[ncol, :]
        L = path.L[ncol, :]
        N = path.N[ncol, :]
        p_eq = path.p_eq[ncol, :]
        p_int = path.p_int[ncol, :]
        p_k = path.p_k[ncol, :]
        p_share = path.p_share[ncol, :]
        Pi_increase = path.Pi_increase[ncol, :]
        Pi_w_increase = path.Pi_w_increase[ncol, :]
        Pi_w = path.Pi_w[ncol, :]
        Pi = path.Pi[ncol, :]
        psi = path.psi[ncol, :]
        q = path.q[ncol, :]
        Q = path.Q[ncol, :]
        qB = path.qB[ncol, :]
        r = path.r[ncol, :]
        ra = path.ra[ncol, :]
        rk = path.rk[ncol, :]
        rl = path.rl[ncol, :]
        s_w = path.s_w[ncol, :]
        s = path.s[ncol, :]
        tau = path.tau[ncol, :]
        valuation_res = path.valuation_res[ncol, :]
        w_res = path.w_res[ncol, :]
        w = path.w[ncol, :]
        Y = path.Y[ncol, :]
        Z = path.Z[ncol, :]
        C_hh = path.C_hh[ncol, :]
        L_hh = path.L_hh[ncol, :]
        A_hh = path.A_hh[ncol, :]
        UCE_hh = path.UCE_hh[ncol, :]

        #################
        # implied paths #
        #################

        ###
        # a. Production Block
        ###

        # inputs: r,w,Y
        # outputs: D,N,I,s

        # Ip -> I and K
        Ip_lag = lag(ini.Ip, Ip)
        I[:] = Ip_lag
        for t in range(par.T):
            K_lag = K[t - 1] if t > 0 else ini.K
            K[t] = (1 - par.delta_K) * K_lag + I[t]

        N[:] = (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
        s[:] = w * N / Y / (1 - par.alpha)
        rk[:] = s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)

        # Q
        for t_ in range(par.T):
            t = par.T - 1 - t_
            Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
            r_plus = r[t + 1] if t < par.T - 1 else ss.r
            rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk

            Q_t = 1.0 / (1.0 + r_plus) * (rk_plus2 + (1.0 - par.delta_K) * Q_plus)
            valuation_res[t] = Q_t - Q[t]

        # investment residual
        for t in range(par.T):  # par.T-1
            Ip_plus = Ip[t + 1] if t < par.T - 1 else ss.I
            r_plus = r[t + 1] if t < par.T - 1 else ss.r

            S = par.phi_K / 2 * (Ip[t] / I[t] - 1.0) ** 2
            Sderiv = par.phi_K * (Ip[t] / I[t] - 1.0)
            Sderiv_plus = par.phi_K * (Ip_plus / Ip[t] - 1.0)

            LHS = 1.0 + S + Ip[t] / I[t] * Sderiv
            RHS = Q[t] + 1.0 / (1.0 + r_plus) * (Ip_plus / Ip[t]) ** 2 * Sderiv_plus
            invest_res[t] = RHS - LHS

        ###
        # b. NKPC prices block
        ###
        # input: s
        # output: Pi
        for t_ in range(par.T):
            t = (par.T - 1) - t_
            Pi_increase_plus = Pi_increase[t + 1] if t < par.T - 1 else 0

            kappa = (1 - par.xi_p) * (1 - par.xi_p / (1 + r[t])) / par.xi_p \
                    * par.e_p / (par.v_p + par.e_p - 1)
            Pi_increase[t] = kappa * (s[t] - (par.e_p - 1) / par.e_p) + (1 / (1 + r[t])) * Pi_increase_plus

        for t in range(par.T):
            Pi_lag = Pi[t - 1] if t > 0 else ini.Pi
            Pi[t] = Pi_increase[t] + Pi_lag

        ###
        # c. Taylor rule block
        ###

        # inputs: Pi
        # outputs: i
        assert par.taylor in ['additive', 'multiplicative', 'simple', 'linear'], 'Taylor rule not implemented'
        for t in range(par.T):
            i_lag = i[t - 1] if t > 0 else ini.i
            Pi_lag = Pi[t - 1] if t > 0 else ini.Pi
            if par.taylor == 'additive':
                i[t] = par.rho_m * i_lag + (1 - par.rho_m) * (ss.r + par.phi_pi * Pi[t]) + em[t]
            elif par.taylor == 'multiplicative':
                i[t] = (1 + ss.r) ** (1 - par.rho_m) * (1 + i_lag) ** par.rho_m \
                       * (1 + Pi[t]) ** ((1 - par.rho_m) * par.phi_pi) * (1 + em[t]) - 1
            elif par.taylor == 'simple':
                i[t] = ss.r + par.phi_pi * Pi[t] + em[t]
        if par.taylor == 'linear':  # constant real interest rate
            for t_ in range(par.T):
                t = (par.T - 1) - t_
                Pi_plus = Pi[t + 1] if t < par.T - 1 else ss.Pi
                i[t] = (1 + ss.r) * (1 + Pi_plus) - 1


        ###
        # d. Finance block
        ###

        # inputs: Div, r
        # outputs: q, rl, ra
        r_lag = lag(ini.r, r)
        rl[:] = r_lag - par.xi

        # Dividends
        I_lag = lag(ini.I, I)
        S = par.phi_K / 2 * (I / I_lag - 1.0) ** 2
        psi[:] = I * S

        Div[:] = Y - w * N - I - psi
        Div_k[:] = rk * K - I - psi
        Div_int[:] = (1 - s) * Y
        # Div_int[:] = Y - w * N - rk * K
        assert np.max(Div - Div_int - Div_k) < 1e-10

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
            p_k[t] = (p_k_plus + Div_k_plus) / (1 + r[t])

            # p_int
            Div_int_plus = Div_int[t + 1] if t < par.T - 1 else ss.Div_int
            p_int_plus = p_int[t + 1] if t < par.T - 1 else ss.p_int
            p_int[t] = (p_int_plus + Div_int_plus) / (1 + r[t])

        assert np.max(p_eq - p_int - p_k) < 1e-10

        A_lag = ini.A_hh
        term_L = (1 + rl[0]) * ini.L_hh + par.xi * ini.L_hh

        term_B = (1 + par.delta_q * q[0]) * ini.B
        term_eq = p_eq[0] + Div[0]

        ra[0] = (term_B + term_eq - term_L) / A_lag - 1
        ra[1:] = r[:-1]

        # agrgegate illiquid assets
        for t in range(par.T):
            A_lag = A[t - 1] if t > 0 else ini.A
            d = ss.ra / (1 + ss.ra) * (1 + ra[t]) * A_lag + par.chi * ((1 + ra[t]) * A_lag - (1 + ss.ra) * par.A_target)
            A[t] = (1 + ra[t]) * A_lag - d


        ###
        # e. Fiscal block
        ###

        # Inputs: q, w, eg
        # Outputs: tau, Z, G

        G[:] = ss.G * (1 + eg)
        for t in range(par.T):
            B_lag = B[t - 1] if t > 0 else ini.B
            # tau without fiscal policy shock
            tau_no_shock = par.phi_tau * ss.q * (B_lag - ss.B) / ss.Y + ss.tau
            # changes of tax rate depending on the shock and fraction of tax financing
            delta_tau = ((1 - par.phi_G) * ss.G * eg[t]) / w[t] / N[t]
            tau[t] = delta_tau + tau_no_shock
            # gov bonds without shock
            B_no_shock = (ss.G + (1 + par.delta_q * q[t]) * B_lag - tau_no_shock * w[t] * N[t]) / q[t]
            # changes in B depending on shock and fraction of tax financing
            delta_B = par.phi_G * ss.G * eg[t] / q[t]
            B[t] = delta_B + B_no_shock
            qB[t] = q[t] * B[t]
            Z[t] = (1 - tau[t]) * w[t] * N[t]
            Z[t] += ez[t]


@nb.njit
def block_post(par, ini, ss, path, ncols=1):
    """ evaluate transition path - after household block """

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol, :]
        B = path.B[ncol, :]
        clearing_A = path.clearing_A[ncol, :]
        clearing_L = path.clearing_L[ncol, :]
        clearing_Y = path.clearing_Y[ncol, :]
        Div_int = path.Div_int[ncol, :]
        Div_k = path.Div_k[ncol, :]
        Div = path.Div[ncol, :]
        eg = path.eg[ncol, :]
        em = path.em[ncol, :]
        ez = path.ez[ncol, :]
        fisher_res = path.fisher_res[ncol, :]
        G = path.G[ncol, :]
        i = path.i[ncol, :]
        I = path.I[ncol, :]
        invest_res = path.invest_res[ncol, :]
        Ip = path.Ip[ncol, :]
        K = path.K[ncol, :]
        L = path.L[ncol, :]
        N = path.N[ncol, :]
        p_eq = path.p_eq[ncol, :]
        p_int = path.p_int[ncol, :]
        p_k = path.p_k[ncol, :]
        p_share = path.p_share[ncol, :]
        Pi_increase = path.Pi_increase[ncol, :]
        Pi_w_increase = path.Pi_w_increase[ncol, :]
        Pi_w = path.Pi_w[ncol, :]
        Pi = path.Pi[ncol, :]
        psi = path.psi[ncol, :]
        q = path.q[ncol, :]
        Q = path.Q[ncol, :]
        qB = path.qB[ncol, :]
        r = path.r[ncol, :]
        ra = path.ra[ncol, :]
        rk = path.rk[ncol, :]
        rl = path.rl[ncol, :]
        s_w = path.s_w[ncol, :]
        s = path.s[ncol, :]
        tau = path.tau[ncol, :]
        valuation_res = path.valuation_res[ncol, :]
        w_res = path.w_res[ncol, :]
        w = path.w[ncol, :]
        Y = path.Y[ncol, :]
        Z = path.Z[ncol, :]
        C_hh = path.C_hh[ncol, :]
        L_hh = path.L_hh[ncol, :]
        A_hh = path.A_hh[ncol, :]
        UCE_hh = path.UCE_hh[ncol, :]

        #################
        # implied paths #
        #################

        # wage phillips curve
        kappa_w = (1 - par.xi_w) * (1 - par.xi_w * par.beta_mean) / par.xi_w \
                  * par.e_w / (par.v_w + par.e_w - 1)

        for t_ in range(par.T):
            t = (par.T - 1) - t_
            Pi_w_increase_plus = Pi_w_increase[t + 1] if t < par.T - 1 else ss.Pi_w - ss.Pi

            s_w[t] = par.nu * N[t] ** (1 / par.frisch) / ((1 - tau[t]) * w[t] * UCE_hh[t])

            Pi_w_increase[t] = kappa_w * (s_w[t] - (par.e_w - 1) / par.e_w) + par.beta_mean * Pi_w_increase_plus


        for t in range(par.T):
            Pi_lag = Pi[t - 1] if t > 0 else ss.Pi
            Pi_w[t] = Pi_w_increase[t] + Pi_lag

        # Fisher equation
        Pi_plus = lead(Pi, ss.Pi)
        fisher_res[:] = 1 + i - (1 + r) * (1 + Pi_plus)

        # wage residual (approximate)
        w_lag = lag(ini.w, w)
        w_res[:] = np.log(w / w_lag) - (Pi_w- Pi)

        # market clearing
        L_hh_lag = lag(ini.L_hh, L_hh)

        clearing_Y[:] = Y - (C_hh + G + I + psi + par.xi * L_hh_lag)
        clearing_A[:] = A_hh - A
