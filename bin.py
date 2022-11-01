# 13.09 solver for investment

def adj_costs(K, K_plus, phi, delta_K):
    I = K_plus - (1 - delta_K) * K
    adj_costs = phi / 2 * (I / K - delta_K) ** 2 * K
    adj_costs_deriv1 = phi * (I / K  - delta_K)
    adj_costs_deriv2 = phi / K
    return adj_costs, adj_costs_deriv1, adj_costs_deriv2

def inv_eq(Q, K_lag, K, K_plus, K_plus2, K_plus3, r_plus, delta_K, phi_K):
    S, S1, _ = adj_costs(K_plus, K_plus2, phi_K, delta_K)
    _, S1_plus, _ = adj_costs(K_plus2, K_plus3, phi_K, delta_K)
    I = K - (1 - delta_K) * K_lag
    I_plus = K_plus - (1 - delta_K) * K
    I_plus2 = K_plus2 - (1 - delta_K) * K_plus
    LHS = 1 + S + I_plus / I * S1
    RHS = Q + (1 / (1 + r_plus)) * (I_plus2 / I_plus) ** 2 * S1_plus
    return LHS - RHS

def residual(x, kwargs_dict):
    """ residual function to optimize using the broyden solver

        :arg x: flattened np.array containing the unknowns"""

    # unpack
    par, ss, Y, w, r = unpack_kwargs(kwargs_dict)

    K, Q = flat_to_K_Q(x)  # back out the unknows from the flattened array
    # as K is fixed for t=0 and t=1 set these to ss value
    K = np.concatenate((np.array([ss.K, ss.K]), K))
    Q = np.concatenate((np.array([ss.Q, ss.Q]), Q))
    # init
    target1 = np.empty_like(Q)
    target2 = np.empty_like(K)

    # labor block
    N = (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
    s = w * N / Y / (1 - par.alpha)
    rk = s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)

    # calculate values for target equation
    for t_ in range(par.T):
        t = (par.T - 1) - t_

        K_lag = K[t - 1] if t > 0 else ss.K
        K_plus = K[t + 1] if t < par.T - 1 else ss.K
        K_plus2 = K[t + 2] if t < par.T - 2 else ss.K
        K_plus3 = K[t + 3] if t < par.T - 3 else ss.K
        Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
        r_plus = r[t + 1] if t < par.T - 1 else ss.r
        rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk
        N_plus2 = N[t + 2] if t < par.T - 2 else ss.N
        s_plus2 = s[t + 2] if t < par.T - 2 else ss.s
        rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk

        # calculate targets
        Q_t = (1 / (1 + r_plus)) * (rk_plus2 + (1 - par.delta_K) * Q_plus)
        target1[t] = Q[t] - Q_t
        target2[t] = inv_eq(Q_t, K_lag, K[t], K_plus, K_plus2, K_plus3, r_plus, par.delta_K, par.phi_K)

    # return target1, target2
    return np.array((target1, target2)).reshape(-1)

def flat_to_K_Q(x):
    nx = x.shape[0]
    assert nx%2 == 0.0
    nx_half = int(nx/2)
    return x[:nx_half], x[nx_half:]


# could not get rid of
# "TypingError: Failed in nopython mode pipeline (step: convert make_function into JIT functions)
# Cannot capture the non-constant value associated with variable 'Y' in a function that will escape."
# "AssertionError: Failed in nopython mode pipeline (step: inline calls to locally defined closures)"

def unpack_kwargs(d):
    return d['par'], d['ss'], d['Y'], d['w'], d['r']

def broyden_solver_cust(f, x0, kwargs_dict=None, jac=None,
                        tol=1e-8, max_iter=100, backtrack_fac=0.5, max_backtrack=30,
                        do_print=False):
    """ numerical solver using the broyden method """

    # a. initial
    x = x0.ravel()
    y = f(x, kwargs_dict)

    if len(x) < len(y):
        print(f"Dimension of x, {len(x)} is less than dimension of y, {len(y)}."
              f" Using least-squares criterion to solve for approximate root.")

    # b. iterate
    for it in range(max_iter):

        # i. current difference
        abs_diff = np.max(np.abs(y))
        if do_print:
            print(f' it = {it:3d} -> max. abs. error = {abs_diff:8.2e}')

        if abs_diff < tol: return x

        # init jac of neccessary
        if not isinstance(jac, np.ndarray):
            if jac == None and kwargs_dict == None:
                # initialize J with Newton!
                jac = obtain_J(f, x, y)
            elif jac == None:
                jac = obtain_J(f, x, y, kwargs_dict)

        # ii. new x
        if len(x) == len(y):
            dx = np.linalg.solve(jac, -y)
        elif len(x) < len(y):
            dx = np.linalg.lstsq(jac, -y, rcond=None)[0]
        else:
            raise ValueError(f"Dimension of x, {len(x)} is greater than dimension of y, {len(y)}."
                             f" Cannot solve underdetermined system.")

        # iii. evalute with backtrack
        for _ in range(max_backtrack):

            try:  # evaluate
                ynew = f(x + dx, kwargs_dict)
                if np.any(np.isnan(ynew)): raise ValueError('found nan value')
            except Exception as e:  # backtrack
                if do_print: print(f'backtracking...')
                dx *= backtrack_fac
            else:  # update jac and break from backtracking
                dy = ynew - y
                jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx) ** 2), dx)
                y = ynew
                x += dx
                break

        else:

            raise ValueError(f'GEModelTools: Number of backtracks exceeds {max_backtrack}')

    else:

        raise ValueError(f'GEModelTools: No convergence after {max_iter} iterations with broyden_solver(tol={tol:.1e})')

def obtain_J(f, x, y, kwargs_dict=None, h=1E-5):
    """Finds Jacobian f'(x) around y=f(x)"""
    nx = x.shape[0]
    ny = y.shape[0]
    J = np.empty((ny, nx))

    for i in range(nx):
        dx = h * (np.arange(nx) == i)
        if kwargs_dict == None:
            J[:, i] = (f(x + dx) - y) / h
        else:
            J[:, i] = (f(x + dx, kwargs_dict) - y) / h
    return J


# for calling in jupyter
ss = model.ss
par = model.par


Y = model.path.Y[0]
w = model.path.w[0]
r = model.path.r[0]

initQ = np.empty_like(Y)
initK = np.empty_like(Y)
initK[:] = model.ini.K
initQ[:] = model.ini.Q
initK = initK[2:] + 0.1   # K in t=0,1 fixed
initQ = initQ[2:] + 0.1

f_args = {'par': par,
          'ss': ss,
          'Y': Y,
          'w': w,
          'r': r}
x0 = np.array([initK, initQ]).ravel()
y0 = residual(x0, kwargs_dict=f_args)
jac = obtain_J(residual, x0, y0, kwargs_dict=f_args)


x_end = broyden_solver_cust(residual ,x0, kwargs_dict = f_args ,jac=None,
    tol=1e-8,max_iter=200,backtrack_fac=0.5,max_backtrack=100,
    do_print=True)

# back out K and Q
K, Q = flat_to_K_Q(x_end)
K = np.concatenate((np.array([ss.K,ss.K]), K))  # append fixed ss values for t=0,1
Q = np.concatenate((np.array([ss.Q,ss.Q]), Q))

I = np.empty_like(Y)

# back out Investment
for t in range(par.T):
    if t == 0:
        I[t] = ss.I
    elif t >= par.T - 1:
        I[t] = ss.K - (1 - par.delta_K) * K[t]
    else:
        I[t] = K[t+1] - (1 - par.delta_K) * K[t]

# forward endo without returns

@nb.njit(parallel=True)
def simulate_hh_forwards_endo_1d(D, i, w, Dbeg_plus):
    """ forward simulation with 1d distribution """
    Nfix = D.shape[0]
    Nz = D.shape[1]
    Nendo1 = D.shape[2]

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):

            Dbeg_plus[i_fix, i_z, :] = 0.0
            for i_endo in range(Nendo1):
                # i. from
                D_ = D[i_fix, i_z, i_endo]

                # ii. to
                i_ = i[i_fix, i_z, i_endo]
                w_ = w[i_fix, i_z, i_endo]
                Dbeg_plus[i_fix, i_z, i_] += D_ * w_
                Dbeg_plus[i_fix, i_z, i_ + 1] += D_ * (1.0 - w_)


@nb.njit(parallel=True)
def simulate_hh_forwards_endo_2d_1iw(D, i, w, Dbeg_plus):
    """ forward simulation with 2d distribution but only along one grid dimension """
    Nfix = D.shape[0]
    Nz = D.shape[1]
    Nendo1 = D.shape[2]
    Nendo2 = D.shape[3]

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            Dbeg_plus[i_fix, i_z, :, :] = 0.0
            for i_endo2 in nb.prange(Nendo2):
                for i_endo1 in nb.prange(Nendo1):
                    # i. from
                    D_ = D[i_fix, i_z, i_endo1, i_endo2]

                    # ii. to
                    i_ = i[i_fix, i_z, i_endo1, i_endo2]
                    w_ = w[i_fix, i_z, i_endo1, i_endo2]
                    Dbeg_plus[i_fix, i_z, i_, i_endo2] += D_ * w_
                    Dbeg_plus[i_fix, i_z, i_ + 1, i_endo2] += D_ * (1.0 - w_)


@nb.njit(parallel=True)
def simulate_hh_forwards_endo_2d_2iw(D, i, w, Dbeg_plus):
    """ forward simulation with 2d distribution along both grid dimension """
    Nfix = D.shape[0]
    Nz = D.shape[1]
    Nendo1 = D.shape[2]
    Nendo2 = D.shape[3]

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            Dbeg_plus[i_fix, i_z, :, :] = 0.0
            for i_endo1 in nb.prange(Nendo1):
                for i_endo2 in nb.prange(Nendo2):
                    # i. from
                    D_ = D[i_fix, i_z, i_endo1, i_endo2]

                    # ii. to
                    i_1_ = i[0, i_fix, i_z, i_endo1, i_endo2]
                    i_2_ = i[1, i_fix, i_z, i_endo1, i_endo2]
                    w_1_ = w[0, i_fix, i_z, i_endo1, i_endo2]
                    w_2_ = w[1, i_fix, i_z, i_endo1, i_endo2]

                    Dbeg_plus[i_fix, i_z, i_1_, i_2_] += w_1_ * w_2_ * D_
                    Dbeg_plus[i_fix, i_z, i_1_ + 1, i_2_] += (1 - w_1_) * w_2_ * D_
                    Dbeg_plus[i_fix, i_z, i_1_, i_2_ + 1] += w_1_ * (1 - w_2_) * D_
                    Dbeg_plus[i_fix, i_z, i_1_ + 1, i_2_ + 1] += (1 - w_1_) * (1 - w_2_) * D_

    # return Dbeg_plus

# could adjust code for residual_with_linear_continuation if seperate bounds should be implemented
# also the bounds are for the targets and not K and Q directly?
opti_bounds = {}

if not opti_bounds:
    x_end = broyden_solver_cust(residual, x0, kwargs_dict=f_args, jac=jac,
                                tol=1e-8, max_iter=200, backtrack_fac=0.5, max_backtrack=100,
                                do_print=False)
else:
    constraint_residual = residual_with_linear_continuation(residual, opti_bounds, kwargs_dict=f_args)
    x_end = broyden_solver_cust(constraint_residual, x0, kwargs_dict=f_args, jac=jac,
                                tol=1e-8, max_iter=200, backtrack_fac=0.5, max_backtrack=100,
                                do_print=False)


# disbale jit

# in notebook
# jit needs to be disabled before numba is imported, so this whole part has to be moved upwards
# if replication_settings['disable_jit']: # disables numba decorator. Needs to be done before numba is imported.
#     os.environ['NUMBA_DISABLE_JIT'] = '1'
# else:
#     os.environ['NUMBA_DISABLE_JIT'] = '0'

# in GEMODELCLASS
# if os.environ.get('NUMBA_DISABLE_JIT') == 1:
#     from .replaced_functions import no_jit as jit   # no numba compiling
# else:
#     from EconModel import jit

#### residual function


@nb.njit
def adj_costs(K, K_plus, phi, delta_K):
    I = K_plus - (1 - delta_K) * K
    adj_costs = phi / 2 * (I / K - delta_K) ** 2 * K
    adj_costs_deriv1 = phi * (I / K  - delta_K)
    adj_costs_deriv2 = phi / K
    return adj_costs, adj_costs_deriv1, adj_costs_deriv2


@nb.njit
def inv_eq(Q, K, K_plus, K_plus2, K_plus3, r_plus, delta_K, phi_K):
    S, S1, _ = adj_costs(K_plus, K_plus2, phi_K, delta_K)
    _, S1_plus, _ = adj_costs(K_plus2, K_plus3, phi_K, delta_K)
    I = K_plus - (1 - delta_K) * K
    I_plus = K_plus2 - (1 - delta_K) * K_plus
    I_plus2 = K_plus3 - (1 - delta_K) * K_plus2
    # S, S1, _ = adj_costs2(I_plus, I, phi_K, delta_K)
    # _, S1_plus, _ = adj_costs2(I_plus2, I_plus, phi_K, delta_K)
    LHS = 1 + S + I_plus / I * S1
    RHS = Q + (1 / (1 + r_plus)) * (I_plus2 / I_plus) ** 2 * S1_plus
    inv_target = LHS - RHS
    # for numerical stability. Otherwise dx in solving the jacobian gets out of hand
    if abs(inv_target) < 1e-8:
        inv_target = 0
    return inv_target

@nb.njit
def residual(x, kwargs_dict):
    """ residual function to optimize using the broyden solver

        :arg x: flattened np.array containing the unknowns"""

    # unpack
    par, ss, Y, w, r, t_predet = unpack_kwargs(kwargs_dict)

    # get values for K and Q. For predetermined values take ss values
    K, Q = flat_to_K_Q(x, t_predet, par.T, ss)  # back out the unknows from the flattened array

    # init target arrays
    target1 = np.empty_like(Q)
    target2 = np.empty_like(K)

    # labor block
    N = np.empty_like(K)
    s = np.empty_like(K)
    rk = np.empty_like(K)
    N[:] = (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
    s[:] = w * N / Y / (1 - par.alpha)
    rk[:] = s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)

    # calculate values for target equation

    for t in range(par.T):

        # K_lag = K[t - 1] if t > 0 else ss.K
        K_plus = K[t + 1] if t < par.T - 1 else ss.K
        K_plus2 = K[t + 2] if t < par.T - 2 else ss.K
        K_plus3 = K[t + 3] if t < par.T - 3 else ss.K
        Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
        r_plus = r[t + 1] if t < par.T - 1 else ss.r
        rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk


        # calculate targets
        Q_t = (1 / (1 + r_plus)) * (rk_plus2 + (1 - par.delta_K) * Q_plus)
        target1[t] = Q[t] - Q_t
        # Capital in t=0,1 fixed
        target2[t] = inv_eq(Q[t], K[t], K_plus, K_plus2, K_plus3, r_plus, par.delta_K, par.phi_K)

    # target values in T-1 always statisfied
    return np.hstack((target2[:-1], target1[t_predet['Q']:-1]))



@nb.njit
def flat_to_K_Q(x, t_predet, T, ss):
    """ Flat array into seperate arrays for K and Q"""
    # old: array had the same length
    # nx = x.shape[0]
    # assert nx%2 == 0.0
    # nx_half = int(nx/2)
    # return x[:nx_half], x[nx_half:]
    K = x[:T - t_predet['K']]
    K = np.concatenate((np.repeat(ss.K, t_predet['K']), K))
    Q = x[T - t_predet['K']:]
    if t_predet['Q'] < 0:
        Q = np.concatenate((Q, np.repeat(ss.Q, abs(t_predet['Q']))))
    else:
        Q = np.concatenate((np.repeat(ss.Q, t_predet['Q']), Q))
    return K, Q

@nb.njit
def residual4(x, kwargs_dict):
    """ residual function to optimize using the broyden solver

        :arg x: flattened np.array containing the unknowns"""

    # unpack
    par, ss, Y, w, r = unpack_kwargs3(kwargs_dict)

    # get values for K and Q. For predetermined values take ss values
    I_frac = x[:par.T]
    Q = x[par.T:]

    # init arrays
    K = np.empty(par.T)
    I = np.empty(par.T)
        # targets
    target1 = np.empty(par.T)
    target2 = np.empty(par.T)
        # labor block
    N = np.empty(par.T)
    s = np.empty(par.T)
    rk = np.empty(par.T)

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
            pass

    # predetermined values
    assert abs(I[0] - ss.I) < 1e-12, 'I[0] != ss.I'

    # caculate labor block
    N[:] = (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
    s[:] = w * N / Y / (1 - par.alpha)
    rk[:] = s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)

    # calculate values for target equation
    for t in range(par.T):
        Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
        r_plus = r[t + 1] if t < par.T - 1 else ss.r
        rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk
        K_plus2 = K[t + 2] if t < par.T - 2 else ss.K
        Y_plus2 = Y[t + 2] if t < par.T - 2 else ss.Y
        w_plus2 = w[t + 2] if t < par.T - 2 else ss.w
        I_frac_plus =  I_frac[t + 1] if t < par.T - 1 else 1

        target1[t], Q_t = inv_eq4(I_frac[t], I_frac_plus,
            Q_plus, r[t], r_plus,
            K_plus2, rk_plus2, Y_plus2, w_plus2,
            par.delta_K, par.phi_K, par.alpha, par.Theta)

        target2[t] = Q[t] - Q_t

    print(target1)
    print(target2)

    return np.hstack((target1, target2))


@nb.njit
def inv_eq4(I_frac, I_frac_plus,
            Q_plus, r, r_plus,
            K_plus2, rk_plus2, Y_plus2, w_plus2,
            delta_K, phi_K, alpha, Theta):
    S, S1, _ = adj_costs3(I_frac, phi_K, delta_K)
    S_plus, S1_plus, _ = adj_costs3(I_frac_plus, phi_K, delta_K)


    deriv_adjplus_kplus2 = - (1 - delta_K) * (1 + S_plus + I_frac_plus + S1_plus) - \
                           I_frac_plus ** 2 * S1_plus

    deriv_N_K_plus2 = - (alpha / (1 - alpha)) * K_plus2 ** (1 / (1 - alpha)) * (Y_plus2 / Theta) ** (1 / (1-alpha))
    deriv_s_N_plus2 = w_plus2 / Y_plus2 / (1 - alpha)
    deriv_rkplus2_Kplus2 = alpha * Y_plus2 * deriv_s_N_plus2 * deriv_N_K_plus2
    deriv_Dplus2_Kplus2 = rk_plus2 + deriv_rkplus2_Kplus2 * K_plus2 - deriv_adjplus_kplus2

    Q_t = (1 / (1 + r_plus)) * (deriv_Dplus2_Kplus2 + Q_plus)

    deriv_Dplus2_Iplus = rk_plus2 + deriv_rkplus2_Kplus2 * K_plus2 + I_frac_plus ** 2 * S1_plus

    deriv_Dplus_Iplus = -(1 + S + I_frac * S1)
    deriv_pplus_Iplus = (1 / (1 + r_plus)) * (deriv_Dplus2_Iplus + Q_plus * (1-delta_K))

    deriv_p_Iplus = (1 / (1 + r)) * (deriv_Dplus_Iplus + deriv_pplus_Iplus)

    # for numerical stability. Otherwise dx in solving the jacobian gets out of hand
    if abs(deriv_p_Iplus) < 1e-12:
        deriv_p_Iplus = 0
    return deriv_p_Iplus, Q_t


@nb.njit
def adj_costs(I_plus, I, phi, delta_K):
    adj_costs = phi / 2 * (I_plus/I - 1.0) ** 2
    adj_costs_deriv1 = phi * (I_plus/I - 1.0)
    adj_costs_deriv2 = phi
    return adj_costs, adj_costs_deriv1, adj_costs_deriv2

@nb.njit
def inv_eq(Q, I, I_plus, I_plus2, r_plus, delta_K, phi_K):
    S, S1, _ = adj_costs(I_plus, I, phi_K, delta_K)
    _, S1_plus, _ = adj_costs(I_plus2, I_plus, phi_K, delta_K)
    LHS = 1.0 + S + I_plus / I * S1
    RHS = Q + (1.0 / (1.0 + r_plus)) * (I_plus2 / I_plus) ** 2 * S1_plus
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
    par, ss, Y, w, r, t_predet = unpack_kwargs(kwargs_dict)

    # get values for K and Q. For predetermined values take ss values
    I, Q = flat_to_I_Q(x, t_predet, par.T, ss)

    # init arrays
    K = np.empty(par.T)
        # targets
    target1 = np.empty(par.T)
    target2 = np.empty(par.T)
        # labor block
    N = np.empty(par.T)
    s = np.empty(par.T)
    rk = np.empty(par.T)

    assert (len(I) == par.T and len(Q) == par.T), 'Q and I dont have length T'
    # back out K and I
    for t in range(par.T):
        # K[t] = (1 - par.delta_K) * K[t - 1] + I[t - 1] if t > 0 else ss.K
        if t == 0:
            K[t] = ss.K
        # elif t == par.T - 1:
        #     K[t] = ss.K
        else:
            K[t] = (1 - par.delta_K) * K[t - 1] + I[t - 1]

    # predetermined values
    assert abs(I[0] - ss.I) < 1e-12, 'I[0] != ss.I'
    assert abs(I[par.T-1] - ss.I) < 1e-12, 'I_T-1 != ss.I'
    assert abs(Q[par.T-1] - 1) < 1e-12, 'Q_T-1 != 1'
    # assert abs(Q[par.T-2] - 1) < 1e-12, 'Q_T-2 != 1'

    # caculate labor block
    N[:] = (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
    s[:] = w * N / Y / (1 - par.alpha)
    rk[:] = s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)

    # calculate values for target equation
    for t in range(par.T):
        Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
        r_plus = r[t + 1] if t < par.T - 1 else ss.r
        rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk
        I_plus =  I[t + 1] if t < par.T - 1 else ss.I
        I_plus2 = I[t + 2] if t < par.T - 2 else ss.I

        # calculate targets
        Q_t = (1 / (1 + r_plus)) * (rk_plus2 + (1 - par.delta_K) * Q_plus)
        target1[t] = inv_eq(Q[t], I[t], I_plus, I_plus2, r_plus, par.delta_K, par.phi_K)
        target2[t] = Q[t] - Q_t


    # target for Q is always statisfied in t = T-1
    # because r_plus, rk_plus2, Q_plus have steady state values and thus has to be Q[par.T - 1]
    assert abs(target2[par.T - 1]) < 1e-12, 'target for Q_t not statisfied in T-1'
    # assert abs(target2[par.T - 2]) < 1e-12, 'target for Q_t not statisfied in T-2'
    # target for I_t+1/I_t is always statisfied in T-1
    assert abs(target1[par.T - 1]) < 1e-12, 'target for I_t+1/I_t not statisfied in T-1'

    # only return non predetermined target values
    return np.hstack((target1[:-1], target2[:-1]))

# initial values for solver
# init_I = np.empty_like(Y)
# init_Q = np.empty_like(Y)
# init_I[:] = ini.I
# init_Q[:] = ini.Q

# # Because of numba
#     # not working for (potential) empty lists
# t_predet = nb.typed.Dict()
# t_predet['I'] = nb.typed.List((0,par.T-1))
# t_predet['Q'] = nb.typed.List((par.T-1,))
#
# for v in t_predet['I'][::-1]:
#     init_I = np.delete(init_I, v)
# for v in t_predet['Q'][::-1]:
#     init_Q = np.delete(init_Q, v)

# f_args = {'par': par,
#           'ss': ss,
#           'Y': Y,
#           'w': w,
#           'r': r,
#           't_predet': t_predet}

# x0 = np.hstack((init_I, init_Q))
# y0 = residual(x0, kwargs_dict=f_args)
# jac = obtain_J(residual, x0, y0, kwargs_dict=f_args)
# x_end = broyden_solver_cust(residual, x0, kwargs_dict=f_args, jac=jac,
#                             tol=1e-8, max_iter=200, backtrack_fac=0.5, max_backtrack=100,
#                             do_print=False)

# # back out I, K, Q
# I_opt, Q_opt = flat_to_I_Q(x_end, t_predet, par.T, ss)
# Q[:] = Q_opt
# I[:] = I_opt
# for t in range(par.T):
#     if t == 0:
#         K[t] = ss.K
#     else:
#         K[t] = (1 - par.delta_K) * K[t - 1] + I[t - 1]


# helper functions that extend the tools in the GEModelClass package

import time
from copy import deepcopy
import numpy as np
import numba as nb

# from EconModel import jit
# from consav.misc import elapsed
# from consav.linear_interp import binary_search
#
# from GEModelTools.simulate_hh import find_i_and_w_1d_1d
# from GEModelTools.simulate_hh import find_i_and_w_1d_1d_path


@nb.njit
def integrate_marg_util(c, D, z_grid, sigma):
    """ integrate marginal utility multiplied by productivity """

    assert c.ndim == 4 and D.ndim == 4, 'c and D do not have the fitting dimensions'
    assert c.shape == D.shape, 'c and D do not have the same dimensions'

    Nfix = c.shape[0]
    Nz = c.shape[1]
    Nl = c.shape[2]
    Na = c.shape[3]

    int_val = 0.0

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            z_i = z_grid[i_z]
            for i_a in nb.prange(Na):
                for i_l in nb.prange(Nl):
                    int_val = int_val + z_i * c[i_fix, i_z, i_l, i_a]**(-sigma) * D[i_fix, i_z, i_l, i_a]

    return int_val


@nb.njit
def broyden_solver_cust(f, x0, kwargs_dict=None, use_jac=None,
                        tol=1e-8, max_iter=100, backtrack_fac=0.5, max_backtrack=30,
                        do_print=False):
    """ numerical solver using the broyden method """

    # a. initial
    x = x0.ravel()
    y = f(x, kwargs_dict)

    if len(x) < len(y) and do_print:
        print("Dimension of x, is less than dimension of y."
              " Using least-squares criterion to solve for approximate root.")

    # b. iterate
    for it in range(max_iter):

        # i. current difference
        abs_diff = np.max(np.abs(y))
        if do_print:
            print('iteration; max. abs. error')
            print(it, abs_diff)
            # print(' it = {:3d} -> max. abs. error = {:8.2e}'.format(it, abs_diff))

        if abs_diff < tol: return x

        if use_jac == None:
            # initialize J with Newton
            if use_jac == None and kwargs_dict == None:
                raise NotImplementedError('residual function needs kwargs_dict at the moment!')
            elif use_jac == None:
                jac = obtain_J(f, x, y, kwargs_dict)

        # ii. new x
        if len(x) == len(y):
            dx = np.linalg.solve(jac, -y)
        elif len(x) < len(y):
            dx = np.linalg.lstsq(jac, -y)[0]    # not allowed to use rcond=None in Numba
        else:
            raise ValueError("Dimension of x is greater than dimension of y."
                             " Cannot solve underdetermined system.")

        # iii. evalute with backtrack
        for _ in range(max_backtrack):

            try:  # evaluate
                ynew = f(x + dx, kwargs_dict)
                if np.any(np.isnan(ynew)): raise ValueError('found nan value')
            except Exception:  # .. as e:
                if do_print: print('backtracking...')
                dx *= backtrack_fac
            else:  # update jac and break from backtracking
                dy = ynew - y
                jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx) ** 2), dx)
                y = ynew
                x += dx
                break

        else:

            raise ValueError('GEModelTools: Number of backtracks exceeded')

    else:

        raise ValueError('GEModelTools: No convergence of broyden solver in solving for investment')


@nb.njit
def obtain_J(f, x, y, kwargs_dict=None, h=1E-5):
    """Finds Jacobian f'(x) around y=f(x)"""
    nx = x.shape[0]
    ny = y.shape[0]
    J = np.empty((ny, nx))

    for i in range(nx):
        dx = h * (np.arange(nx) == i)
        J[:, i] = (f(x + dx, kwargs_dict) - y) / h
    return J

# taken from sequence-jacobian repository
def residual_with_linear_continuation(residual, bounds, kwargs_dict=None, eval_at_boundary=False,
                                      boundary_epsilon=1e-4, penalty_scale=1e1,
                                      verbose=False):
    """Modify a residual function to implement bounds by an additive penalty for exceeding the boundaries
    provided, scaled by the amount the guess exceeds the boundary.

    e.g. For residual function f(x), desiring x in (0, 1) (so assuming eval_at_boundary = False)
         If the guess for x is 1.1 then we will censor to x_censored = 1 - boundary_epsilon, and return
         f(x_censored) + penalty (where the penalty does not require re-evaluating f() which may be costly)

    residual: `function`
        The function whose roots we want to solve for
    bounds: `dict`
        A dict mapping the names of the unknowns (`str`) to length two tuples corresponding to the lower and upper
        bounds.
    eval_at_boundary: `bool`
        Whether to allow the residual function to be evaluated at exactly the boundary values or not.
        Think of it as whether the solver will treat the bounds as creating a closed or open set for the search space.
    boundary_epsilon: `float`
        The amount to adjust the proposed guess, x, by to calculate the censored value of the residual function,
        when the proposed guess exceeds the boundaries.
    penalty_scale: `float`
        The linear scaling factor for adjusting the penalty for the proposed unknown values exceeding the boundary.
    verbose: `bool`
        Whether to print out additional information for how the constrained residual function is behaving during
        optimization. Useful for tuning the solver.
    """
    lbs = np.asarray([v[0] for v in bounds.values()])
    ubs = np.asarray([v[1] for v in bounds.values()])

    def constr_residual(x, kwargs_dict=None, residual_cache=[]):
        """Implements a constrained residual function, where any attempts to evaluate x outside of the
        bounds provided will result in a linear penalty function scaled by `penalty_scale`.

        Note: We are purposefully using residual_cache as a mutable default argument to cache the most recent
        valid evaluation (maintain state between function calls) of the residual function to induce solvers
        to backstep if they encounter a region of the search space that returns nan values.
        See Hitchhiker's Guide to Python post on Mutable Default Arguments: "When the Gotcha Isn't a Gotcha"
        """
        if eval_at_boundary:
            x_censored = np.where(x < lbs, lbs, x)
            x_censored = np.where(x > ubs, ubs, x_censored)
        else:
            x_censored = np.where(x < lbs, lbs + boundary_epsilon, x)
            x_censored = np.where(x > ubs, ubs - boundary_epsilon, x_censored)

        residual_censored = residual(x_censored, kwargs_dict)

        if verbose:
            print(f"Attempted x is {x}")
            print(f"Censored x is {x_censored}")
            print(f"The residual_censored is {residual_censored}")

        if np.any(np.isnan(residual_censored)):
            # Provide a scaled penalty to the solver when trying to evaluate residual() in an undefined region
            residual_censored = residual_cache[0] * penalty_scale

            if verbose:
                print(f"The new residual_censored is {residual_censored}")
        else:
            if not residual_cache:
                residual_cache.append(residual_censored)
            else:
                residual_cache[0] = residual_censored

        if verbose:
            print(f"The residual_cache is {residual_cache[0]}")

        # Provide an additive, scaled penalty to the solver when trying to evaluate residual() outside of the boundary
        residual_with_boundary_penalty = residual_censored + \
                                         (x - x_censored) * penalty_scale * residual_censored
        return residual_with_boundary_penalty

    return constr_residual

# # initial values for solver
# init_Q = np.empty_like(Y)
# init_Q[:] = ini.Q
# init_I_plus = np.empty_like(Y)
# init_I_plus[:] = ini.I
#
# f_args = {'par': par,
#           'ss': ss,
#           'Y': Y,
#           'w': w,
#           'r': r}
#
# x0 = np.hstack((init_I_plus, init_Q))
# y0 = residual(x0, kwargs_dict=f_args)
# jac = obtain_J(residual, x0, y0, kwargs_dict=f_args)
# # with jac=None: calculate jacobian again each iteration
# x_end = broyden_solver_cust(residual, x0, kwargs_dict=f_args, use_jac=jac,
#                             tol=1e-5, max_iter=100, backtrack_fac=0.5, max_backtrack=100,
#                             do_print=False)
#
# # back out I, K, Q
# I_plus[:] = x_end[:par.T]
# Q[:] = x_end[par.T:]

# # investment equations
# for t in range(par.T):
#     Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
#     r_plus = r[t + 1] if t < par.T - 1 else ss.r
#     rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk
#
#     I_t = I[t]
#     I_t_plus = I[t + 1] if t < par.T - 1 else ss.I
#     I_t_plus2 = I[t + 2] if t < par.T - 2 else ss.I
#     I_frac = I_t_plus / I_t
#     I_frac_plus = I_t_plus2 / I_t_plus
#
#     # calculate targets
#     Q_t = (1 / (1 + r_plus)) * (rk_plus2 + (1 - par.delta_K) * Q_plus)
#     invest_res[t] = inv_eq(Q[t], I_frac, I_frac_plus, r_plus, par.delta_K, par.phi_K)
#     valuation_res[t] = Q_t - Q[t]
#
#     if t < 2:
#         k_target = ss.K - K[t]
#         if k_target > 0:
#             if test[t] > 0:
#                 k_res[t] = k_target + test[t]
#             else:
#                 k_res[t] = k_target - test[t]
#         else:
#             if test[t] > 0:
#                 k_res[t] = k_target - test[t]
#             else:
#                 k_res[t] = k_target + test[t]
#     else:
#         k_res[t] = test[t]

@nb.njit
def NKPC_eq(x, par, r, s, Pi_plus):
    gap = s - (par.e_p - 1)/par.e_p

    kappa = (1-par.xi_p) * (1-par.xi_p/(1+r)) / par.xi_p \
            * par.e_p / (par.v_p + par.e_p - 1)

    NKPC = x - 1/(1+r) * kappa * gap - 1/(1+r)*Pi_plus

    return NKPC

# @nb.njit
# def unpack_kwargs(d):
#     """ simple unpacking funtion specific to the current residual function"""
#     return d['par'], d['ss'], d['Y'], d['w'], d['r'], d['t_predet']

@nb.njit
def flat_to_K_Q(x, t_predet, T, ss):
    """ Flat array into seperate arrays for I_t+1/I_t and Q"""
    K = np.empty(T)
    Q = np.empty(T)

    # for numba
    mask_K = np.ones(T,  nb.bool_)
    mask_Q = np.ones(T,  nb.bool_)
    for v in t_predet['K']:
        mask_K[v] = False
    for v in t_predet['Q']:
        mask_Q[v] = False

    K[mask_K] = x[:(T - len(t_predet['K']))]
    K[np.invert(mask_K)] = ss.K

    Q = x[(T - len(t_predet['K'])):]
    # Q[mask_Q] = x[(T - len(t_predet['K'])):]
    # Q[np.invert(mask_Q)] = ss.Q

    return K, Q


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
    # # for numerical stability. Otherwise dx in solving the jacobian gets out of hand
    # if abs(inv_target) < 1e-12:
    #     inv_target = 0
    return inv_target

@nb.njit
def residual(x, kwargs_dict):
    """ residual function to optimize using the broyden solver

        :arg x: flattened np.array containing the unknowns"""

    # unpack
    par, ss, Y, w, r = unpack_kwargs(kwargs_dict)

    # get values for K and Q. For predetermined values take ss values
    # K, Q = flat_to_K_Q(x, t_predet, par.T, ss)
    I_plus = x[:par.T]
    Q = x[par.T:]

    # init arrays
    I_frac = np.empty(par.T)
    I = np.empty(par.T)
        # targets
    target1 = np.empty(par.T)
    target2 = np.empty(par.T)
    target3 = np.empty(par.T)
        # labor block
    N = np.empty(par.T)
    s = np.empty(par.T)
    rk = np.empty(par.T)
    I = np.empty(par.T)
    K = np.empty(par.T)

    # I_plus -> I + K
    for t in range(par.T):
        I[t] = I_plus[t - 1] if t > 0 else ss.I
        K[t] = (1 - par.delta_K) * K[t - 1] + I[t - 1] if t > 0 else ss.K


    # caculate labor block
    N[:] = (Y / (par.Theta * K ** par.alpha)) ** (1 / (1 - par.alpha))
    s[:] = w * N / Y / (1 - par.alpha)
    rk[:] = s * par.alpha * par.Theta * K ** (par.alpha - 1) * N ** (1 - par.alpha)

    # calculate values for target equation
    for t in range(par.T):
        Q_plus = Q[t + 1] if t < par.T - 1 else ss.Q
        r_plus = r[t + 1] if t < par.T - 1 else ss.r
        rk_plus2 = rk[t + 2] if t < par.T - 2 else ss.rk

        I_t = I[t]
        I_t_plus = I_plus[t]
        I_t_plus2 = I_plus[t + 1] if t < par.T - 1 else ss.I

        I_frac = I_t_plus / I_t
        I_frac_plus = I_t_plus2 / I_t_plus

        # calculate targets
        Q_t = (1 / (1 + r_plus)) * (rk_plus2 + (1 - par.delta_K) * Q_plus)
        target1[t] = inv_eq(Q[t], I_frac, I_frac_plus, r_plus, par.delta_K, par.phi_K)
        target2[t] = Q[t] - Q_t
        if t < par.T - 2:
            target3[t] = K[t + 2] - (1 - par.delta_K) * K[t + 1] - I_plus[t]
        elif t == par.T - 2:    # to make sure I_T is s.t. K_T+1=ss.K
            target3[t] = ss.K - (1 - par.delta_K) * K[t + 1] - I_plus[t]
        else:   # to make sure I_T+1 = ss.I
            target3[t] = ss.K - (1 - par.delta_K) * ss.K - I_plus[t]


    return np.hstack((target1, target2, target3))

@nb.njit
def NKPC_w_eq(x, par, s_w, Pi_w_plus):
    gap = s_w - (par.e_w - 1)/par.e_w

    kappa = (1-par.xi_w) * (1-par.xi_w*par.beta_mean) / par.xi_w \
            * par.e_w / (par.v_w + par.e_w - 1)

    NKPC_w = x - kappa * gap - par.beta_mean*Pi_w_plus

    return NKPC_w