import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem
import blocks

class HANKnoStickyModelClass(EconModelClass, GEModelClass):

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
        self.inputs_hh = ['Z','ra','rl']  # direct inputs
        self.inputs_hh_z = []  # transition matrix inputs
        self.outputs_hh = ['c','l','a','uce']  # outputs
        self.intertemps_hh = ['vbeg_l_a']  # intertemporal variables

        # c. GE
        self.shocks = ['eg','em']  # exogenous shocks
        self.unknowns = ['r','w','Y','Ip','Q']  # endogenous unknowns
        self.targets = ['fisher_res','w_res','clearing_Y','invest_res','valuation_res']  # targets = 0

        # d. all variables
        self.varlist = [
            'A',
            'B',
            'clearing_A',
            'clearing_L',
            'clearing_Y',
            'clearing_Y',
            'clearing_fund_start',
            'clearing_fund_end',
            'Div_int',
            'Div_k',
            'Div',
            'eg',
            'em',
            'fisher_res',
            'G',
            'i',
            'I',
            'invest_res',
            'Ip',
            'K',
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
        par.r_ss_target = (1 + r_ss_target_p_a)**(1/4) - 1

        par.K_Y_ratio = 2.23 # capital to GDP
        par.L_Y_ratio = 0.23 # liquid assets to GDP
        par.G_Y_ratio = 0.16 # spending-to-GDP
        par.qB_Y_ratio = 0.46 # government bonds to GDP

        # a. preferences
        par.sigma = 2.0  # CRRA coefficient
        par.beta_mean =  0.97950170  # discount factor, mean, range is [mean-width,mean+width]
        par.beta_delta = 0.00000  # discount factor, width, range is [mean-width,mean+width]
        par.frisch = 0.5 #  Frisch elasticity
        par.nu = np.nan  # disutility from labor
        par.inattention = 0.0

        # b. income and saving parameters
        par.rho_e = 0.966  # AR(1) parameter
        par.sigma_e = 0.50  # std. of e
        par.chi = 0.005*4  # redistribution share for illiquid assets
        par.A_target = np.nan  # illiquid asset target

        # c. intermediate good firms
        par.Theta = np.nan # productivity factor
        par.alpha = np.nan # capital share
        par.delta_K = 0.053 # depreciation of capital

        par.mu_p = 1.06 # mark-up
        par.e_p = par.mu_p/(par.mu_p-1)
        par.xi_p = 0.926 # calvo price stickiness
        par.v_p = 0 # Kimball superelasticity for prices

        # d. capital goods firms
        par.phi_K = 9.0      # (inverse of the) elasticity of investment

        # e. unions
        par.xi_w = 0.899 # calvo wage stickiness
        par.e_w = par.e_p
        par.v_w = 0 # Kimball superelasticity for wages

        # f. central bank
        par.rho_m = 0.6  # Taylor rule intertia
        par.phi_pi = 1.5 # Taylor rule coefficient

        # g. government
        par.phi_tau = 0.1 # response of tax rate to debt
        par.phi_G = 0 # tax financing of government expenditure shock
        maturity = 5*4 # Maturity of government debt
        par.delta_q = (maturity-1)*(1+par.r_ss_target)/maturity

        # h. mutal fund
        xi_p_a = 0.065 # intermedation spread (p.a.)
        par.xi = 1+par.r_ss_target-(1+r_ss_target_p_a-xi_p_a)**(1/4)

        # i. grids
        par.l_min = 0.0 # maximum point in grid for a
        par.l_max = 10.0 # maximum point in grid for a
        par.Nl = 50 # number of grid points

        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 10.0 # maximum point in grid for a
        par.Na = 50 # number of grid points

        # j. shocks
        par.jump_eg = 0.0 # initial jump
        par.rho_eg = 0.0 # AR(1) coefficient
        par.std_eg = 0.0 # std. of innovation
        par.jump_em = 0.00025 # initial jump
        par.rho_em = 0.6 # AR(1) coefficient
        par.std_em = 0.0 # std. of innovation

        # k. misc.
        par.T = 200  # length of transition path
        par.simT = 100  # length of simulation

        par.max_iter_solve = 50_000  # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000  # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100  # maximum number of iteration when solving eq. system

        par.tol_ss = 1e-11  # tolerance when finding steady state
        par.tol_solve = 1e-12  # tolerance when solving household problem
        par.tol_simulate = 1e-12  # tolerance when simulating household problem
        par.tol_broyden = 1e-10  # tolerance when solving eq. system

    def allocate(self):
        """ allocate model """

        par = self.par
        par.beta_grid = np.zeros(par.Nfix)

        self.allocate_GE()

    # override steady state functions
    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss