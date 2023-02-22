import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass
from GEModelTools import tests
from HANKStickyAnalytics import HANKStickyAnalyticsClass

import steady_state
import household_problem
import blocks


#class HANKStickyModelClass(EconModelClass, GEModelClass, HANKStickyAnalyticsClass):
class HANKStickyModelClass(EconModelClass, HANKStickyAnalyticsClass):
    def __init__(self, savefolder='saved', *args, **kwargs):
        EconModelClass.__init__(self, *args, **kwargs)
        self.savefolder = savefolder


    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','sim','ss','path']

        # b. household
        self.grids_hh = ['l','a']  # grids
        self.pols_hh = ['l','a']  # policy functions
        self.inputs_hh = ['tau','wN','ra','rl','ez','eg_transfer']  # direct inputs
        self.inputs_hh_z = []  # transition matrix inputs
        self.outputs_hh = ['c','l','a','uce']  # outputs
        self.intertemps_hh = ['vbeg_l_a']  # intertemporal variables

        # c. GE
        # self.shocks = ['eg','em','ez']  # exogenous shocks
        self.shocks = ['eg','em','ez','eg_direct','eg_distribution','eg_debt','eg_transfer']  # exogenous shocks
        self.unknowns = ['r','w','Y','Ip','rk']  # endogenous unknowns
        self.targets = ['fisher_res','w_res','clearing_Y','invest_res','clearing_K']  # targets = 0

        # d. all variables
        self.varlist = [
            'A',
            'B',
            'MPC_target',
            'hh_wealth',
            'clearing_A',
            'clearing_L',
            'clearing_C',
            'clearing_Y',
            'clearing_K',
            'clearing_wealth',
            'Div_int',
            'Div_k',
            'Div',
            'eg',
            'eg_direct',
            'eg_distribution',
            'eg_debt',
            'eg_transfer',
            'ez',
            'em',
            'fisher_res',
            'G',
            'i',
            'I',
            'invest_res',
            'Ip',
            'K',
            'Kd',
            'L',
            'N',
            'p_eq',
            'p_int',
            'p_k',
            'p_share',
            'Pi_increase',
            'Pi_w_increase',
            'Pi_w',
            'Pi',
            'psi',
            'q',
            'Q',
            'qB',
            'r',
            'ra',
            'rk',
            'rl',
            's_w',
            's',
            'tau',
            'wN',
            'valuation_res',
            'w_res',
            'w',
            'Y',
            'Z',
            ]

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = blocks.block_pre
        self.block_post = blocks.block_post

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1  # number of fixed discrete states (here discount factor)
        par.Nz = 7  # number of stochastic discrete states (here productivity)

        # targets
        r_ss_target_p_a = 0.05
        par.r_ss_target = (1 + r_ss_target_p_a)**(1/4) - 1  # quarterly

        par.MPC_target = 0.525
        par.K_Y_ratio = 2.23*4  # capital to GDP  - quarterly
        # par.L_Y_ratio = 0.23*4  # liquid assets to GDP - quarterly
        # par.L_Y_ratio = np.nan
        par.hh_wealth_Y_ratio = 3.82*4  # aggregate household wealth - quarterly
        par.A_L_ratio = 0.85
        par.G_Y_ratio = 0.16  # spending-to-GDP
        par.qB_Y_ratio = 0.42*4  # government bonds to GDP - quarterly
        # par.A_Y_ratio = (par.hh_wealth_Y_ratio - par.L_Y_ratio)
        # par.A_Y_ratio = np.nan
        par.A_target = np.nan
        assert par.Nfix == 1

        # a. preferences
        par.sigma = 1.0  # CRRA coefficient
        # TODO: change to new quarterly beta
        par.beta_mean = 0.9951   # discount factor, mean, range is [mean-width,mean+width]
        par.beta_delta = 0.00000  # discount factor, width, range is [mean-width,mean+width]
        par.frisch = 0.5  # Frisch elasticity
        par.nu = np.nan   # Disutility from labor
        par.inattention = 0.0

        # b. income and saving parameters
        par.rho_e = 0.966  # AR(1) parameter
        par.sigma_e = 0.5  # 0.5 std. of e # TODO: calibrate to match MPC?
        par.chi = 0.009  # 0.009  # redistribution share for illiquid assets # TODO: calibrate to macth MPC
        par.A_target = np.nan  # illiquid asset target

        # c. intermediate good firms
        par.Theta = np.nan  # productivity factor
        par.alpha = np.nan  # capital share
        par.delta_K = 0.053/4  # depreciation of capital - quarterly

        par.mu_p = np.nan
        par.e_p = np.nan
        par.xi_p = 0.926  # calvo price stickiness
        par.v_p = 10 # *2  # Kimball superelasticity for prices - # TODO: calibrate?

        # d. capital goods firms
        par.phi_K = 9.0  # 3.0 (inverse of the) elasticity of investment    # TODO: calibrate

        # e. unions
        par.xi_w = 0.899 #0.899  # calvo wage stickiness
        par.e_w = np.nan
        par.v_w = 10 # *2  # Kimball superelasticity for wages # TODO: calibrate?

        # f. central bank
        par.rho_m = 0.89  # Taylor rule intertia    # TODO: estimate?
        par.phi_pi = 1.25  # Taylor rule coefficient # TODO: estimate?

        # g. government
        par.phi_tau = 0.1 # response of tax rate to debt # TODO: calibrate (at least sensitivity analysis)
        par.phi_G = 0.5 # deficit financing of government expenditure shock
        maturity = 5*4 # Maturity of government debt
        par.delta_q = (maturity-1)*(1+par.r_ss_target)/maturity
        par.taylor = 'additive'  # 'multiplicative', 'additive' or 'simple' for different taylor rules

        # h. mutual fund
        xi_p_a = 0.065  # intermediation spread (p.a.)
        par.xi = 1+par.r_ss_target-(1+r_ss_target_p_a-xi_p_a)**(1/4)  # quarterly

        # i. grids
        par.l_min = 0.0  # maximum point in grid for a
        par.l_max = 10.0*3  # maximum point in grid for a  - quarterly
        par.Nl = 100  # number of grid points

        par.a_min = 0.0  # maximum point in grid for a
        par.a_max = 10.0*3  # maximum point in grid for a - quarterly
        par.Na = 100  # number of grid points

        # j. shocks
        # 1. fiscal policy shock
        par.jump_eg = 0.01  # initial jump
        par.rho_eg = 0.9  # AR(1) coefficient
        par.std_eg = 0.0  # std. of innovation
        # 2. monetary policy
        par.jump_em = 0.0 # initial jump
        par.rho_em = 0.0  # AR(1) coefficient
        par.std_em = 0.0  # std. of innovation
        # 3. persistent income shock
        par.jump_ez = 0.0  # initial jump
        par.rho_ez = 0.6  # AR(1) coefficient
        par.std_ez = 0.0  # std. of innovation
        # # 4. fiscal policy shocks for decomposition
        # par.decomp_distribution = False
        # par.decomp_debt = False
        # par.decomp_direct = False
        # 4a. direct effect
        par.jump_eg_direct = par.jump_eg  # initial jump
        par.rho_eg_direct = par.rho_eg  # AR(1) coefficient
        par.std_eg_direct = par.std_eg  # std. of innovation
        # 4b. distributional effect
        par.jump_eg_distribution = par.jump_eg  # initial jump
        par.rho_eg_distribution = par.rho_eg  # AR(1) coefficient
        par.std_eg_distribution = par.std_eg  # std. of innovation
        # 4c. crowding out effect
        par.jump_eg_debt = par.jump_eg  # initial jump
        par.rho_eg_debt = par.rho_eg  # AR(1) coefficient
        par.std_eg_debt = par.std_eg  # std. of innovation
        # 4d. direct transfers
        par.jump_eg_transfer = 0.0  # initial jump
        par.rho_eg_transfer = 0.0  # AR(1) coefficient
        par.std_eg_transfer = 0.0  # std. of innovation


        # k. misc.
        par.T = 350  # length of transition path
        par.simT = 100  # length of simulation

        par.max_iter_solve = 50_000  # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000  # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100  # maximum number of iteration when solving eq. system

        par.tol_ss = 1e-11  # tolerance when finding steady state
        par.tol_solve = 1e-11  # tolerance when solving household problem
        par.tol_simulate = 1e-11  # tolerance when simulating household problem
        par.tol_broyden = 1e-11  # tolerance when solving eq. system

        par.start_dbeg_opti = True  # starts with optimal distribution along illiquid asset grid to speed up find_ss
        assert par.Nfix == 1, "For now, par.start_dbeg_opti = True works only without multiple beta"

        par.print_non_lin_warning = True


    def allocate(self):
        """ allocate model """

        par = self.par
        par.beta_grid = np.zeros(par.Nfix)

        self.allocate_GE()


    # override steady state functions
    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

    # additional tests
    test_jacs_sticky = tests.jacs_sticky
