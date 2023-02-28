import numpy as np
import numba as nb

from GEModelTools import lag, lead, prev, next


@nb.njit
def production_firm(par, ini, ss, Y, Ip, I, K, N, s, w, rk, psi,
                    Div, Div_k, Div_int, Q, r):

    Ip_lag = lag(ini.Ip, Ip)
    I[:] = Ip_lag

    for t in range(par.T):
        K_lag = K[t - 1] if t > 0 else ini.K
        K[t] = (1 - par.delta_K) * K_lag + I[t]

    N[:] = (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
    s[:] = w * N / Y / (1 - par.alpha)
    rk[:] = s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)

    # dividends
    I_lag = lag(ini.I, I)
    S = par.phi_K / 2 * (I / I_lag - 1.0) ** 2
    psi[:] = I * S
    Div[:] = Y - w * N - I - psi
    Div_k[:] = rk * K - I - psi
    Div_int[:] = (1 - s) * Y
    assert np.max(Div - Div_int - Div_k) < 1e-10

    # investment residual
    for k in range(par.T):
        t = par.T - 1 - k

        Q_plus = next(Q, t, ss.Q)
        rk_plus2 = next(rk, t+1, ss.rk)

        Q[t] = (rk_plus2 + (1 - par.delta_K) * Q_plus) / (1 + r[t])

@nb.njit
def invest_residual(par, ini, ss, Ip, r, Q, invest_res):
    Ip_lag = lag(ini.Ip, Ip)
    r_plus = lead(r, ss.r)
    Ip_plus = lead(Ip, ss.I)

    Sp = par.phi_K / 2 * (Ip / Ip_lag - 1.0) ** 2
    Spderiv = par.phi_K * (Ip / Ip_lag - 1.0)
    Spderiv_plus = par.phi_K * (Ip_plus / Ip - 1.0)

    LHS = 1.0 + Sp + Ip / Ip_lag * Spderiv
    RHS = Q + 1.0 / (1.0 + r_plus) * (Ip_plus / Ip) ** 2 * Spderiv_plus

    invest_res[:] = RHS - LHS


@nb.njit
def price_setters(par, ini, ss, s, Pi):
    for t in range(par.T):
        Pi_lag = prev(Pi, t, ini.Pi)
        d_s = 0.0
        for k in range(t, par.T):
            d_s += (1 / (1 + ss.r)) ** (k-t) * (s[k] - (par.e_p - 1) / par.e_p)
        Pi[t] = par.kappa * d_s + Pi_lag


@nb.njit
def taylor(par, ini, ss, em, Pi, i):
    for t in range(par.T):
        i_lag = prev(i, t, ini.i)
        i[t] = par.rho_m * i_lag + (1 - par.rho_m) * (ss.r + par.phi_pi * Pi[t]) + em[t]

# @nb.njit
# def taylor_passive(par, ini, ss, i, em):
#     i[:] = ss.i + em

@nb.njit
def taylor_constant_r(par, ini, ss, Pi, i, em):
    Pi_plus = lead(Pi, ss.Pi)
    i[:] = (1 + ss.r) * (1 + Pi_plus) - 1 + em

@nb.njit
def fisher(par, ini, ss, Pi, i, r, fisher_res):
    Pi_plus = lead(Pi, ss.Pi)
    fisher_res[:] = 1 + i - (1 + r) * (1 + Pi_plus)

@nb.njit
def mutual_fund(par, ini, ss, r, Div, q, p_eq, rl, ra):

    r_lag = lag(ini.r, r)
    rl[:] = r_lag - par.xi

    for t_ in range(par.T):
        t = (par.T - 1) - t_

        # q
        q_plus = next(q, t, ss.q)
        q[t] = (1 + par.delta_q * q_plus) / (1 + r[t])

        # p_eq
        p_eq_plus = next(p_eq, t, ss.p_eq)
        Div_plus = next(Div, t, ss.Div)
        p_eq[t] = (Div_plus + p_eq_plus) / (1 + r[t])

    A_lag = ini.A
    term_L = (1 + rl[0]) * ini.L + par.xi * ini.L

    term_B = (1 + par.delta_q * q[0]) * ini.B
    term_eq = p_eq[0] + Div[0]

    ra[0] = (term_B + term_eq - term_L) / A_lag - 1
    ra[1:] = r[:-1]


@nb.njit
def government_custom(par, ini, ss, tau, B, G, eB, eG, etau, Z, w, N):
    tau[:] = ss.tau + etau
    B[:] = ss.B + eB
    G[:] = ss.G + eG

    Z[:] = (1 - tau) * w * N

@nb.njit
def government(par, ini, ss, eg, eg_transfer, w, N, q, G, tau, B, Z):
    G[:] = ss.G * (1 + eg) + eg_transfer
    for t in range(par.T):
        B_lag = prev(B, t, ini.B)

        tau_no_shock = ss.tau + par.phi_tau * ss.q * (B_lag - ss.B) / ss.Y
        delta_tau = (1 - par.phi_G) * (G[t] - ss.G) / (w[t] * N[t])

        tau[t] = tau_no_shock + delta_tau
        B[t] = ((1 + par.delta_q * q[t]) * B_lag + G[t] - tau[t] * w[t] * N[t]) / q[t]

    Z[:] = (1 - tau) * w * N

def government_constant_B(par, ini, ss, eg, eg_transfer, w, N, q, G, tau, B, Z):

    G[:] = ss.G * (1 + eg) + eg_transfer
    B[:] = ss.B
    tau[:] = (G + (1 + par.delta_q * q) * ss.B - q * ss.B) / (w * N)

    Z[:] = (1 - tau) * w * N

# @nb.njit
# def government3(par, ini, ss, eg, w, N, q, G, tau, B, Z):
#
#     q_lag = lag(ini.q, q)
#     w_lag = lag(ini.w, w)
#     N_lag = lag(ini.N, N)
#     d_q = q - q_lag
#     d_w = w - w_lag
#     d_N = N - N_lag
#
#     G[:] = ss.G * (1.0 + eg)
#     for t in range(par.T):
#         B_lag = prev(B, t, ini.B)
#         B_lag2 = prev(B, t-1, ini.B)
#         d_B_lag = B_lag - B_lag2
#         tau_lag = prev(tau, t, ini.tau)
#
#         d_B = par.phi_G * (ss.G * eg[t] + (1 + par.delta_q * d_q[t]) * ss.B + (1 + par.delta_q * q[t]) * d_B_lag
#                            - d_q[t] * ss.B - ss.tau * d_w[t] * d_N[t]) / q[t]
#         d_tau = (1 - par.phi_G) * (ss.G * eg[t] + (1 + par.delta_q * d_q[t]) * ss.B + (1 + par.delta_q * q[t]) * d_B_lag
#                                    - d_q[t] * ss.B - ss.tau * d_w[t] * d_N[t]) / (w[t] * N[t])
#
#         B[t] = B_lag + d_B
#         tau[t] = tau_lag + d_tau
#
#     Z[:] = (1 - tau) * w * N




@nb.njit
def union(par, ini, ss, tau, w, UCE_hh, s_w, N, Pi_w, Pi):
    Pi_lag = lag(ini.Pi, Pi)

    s_w[:] = par.nu * N ** (1 / par.frisch) / ((1 - tau) * w * UCE_hh)

    for t in range(par.T):
        d_s_w = 0.0
        for k in range(t, par.T):
            d_s_w += par.beta_mean ** (k-t) * (s_w[k] - (par.e_w - 1) / par.e_w)
        Pi_w[t] = par.kappa * d_s_w + Pi_lag[t]


@nb.jit
def real_wage(par, ini, ss, w, Pi_w, Pi, w_res):
    w_lag = lag(ini.w, w)
    w_res[:] = np.log(w / w_lag) - (Pi_w - Pi)


@nb.jit
def market_clearing(par, ini, ss, Y, L_hh, C_hh, G, eg_transfer, I, psi, q, B, p_eq, A_hh, qB, A, L, clearing_Y,
                    clearing_A, clearing_L):
    # Y
    L[:] = L_hh
    L_lag = lag(ini.L, L)
    clearing_Y[:] = Y - (C_hh + (G - eg_transfer) + I + psi + par.xi * L_lag)

    # A
    qB[:] = q * B
    A[:] = p_eq + qB - L
    clearing_A[:] = A_hh - A

    # L
    clearing_L[:] = L_hh - L

    # clearing_wealth =

