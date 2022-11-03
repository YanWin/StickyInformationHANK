import sys

import GEModelTools
import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass
from GEModelTools import simulate_hh
from GEModelTools import replaced_functions

import steady_state
import household_problem
import blocks
import helper_functions


class HANKnoStickyModelClass(EconModelClass, GEModelClass):

    def __init__(self, savefolder='saved', *args, **kwargs):
        EconModelClass.__init__(self, *args, **kwargs)
        self.savefolder = savefolder


    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par', 'ini', 'sim', 'ss', 'path']

        # b. household
        self.grids_hh = ['l', 'a']  # grids
        self.pols_hh = ['l', 'a']  # policy functions
        self.inputs_hh = ['Z', 'ra', 'rl']  # direct inputs
        self.inputs_hh_z = []  # transition matrix inputs
        self.outputs_hh = ['c', 'l', 'a', 'uce']  # outputs
        self.intertemps_hh = ['vbeg_l_a']  # intertemporal variables


        # c. GE
        self.shocks = ['eg', 'em']  # exogenous shocks
        self.unknowns = ['r', 'w', 'Y', 'Ip', 'Q', 'Pi', 'Pi_w']  # endogenous unknowns
        self.targets = ['fisher_res', 'w_res', 'clearing_Y', 'invest_res', 'valuation_res', 'NKPC_res', 'NKPC_w_res']  # targets = 0

        # d. all variables
        self.varlist = [
            'r', 'ra','rl', 'i',
            'Pi', 'Pi_w',
            'G', 'tau', 'B',
            'Y', 'N', 'I', 'K', 'Div', 'Q',
            'C', 'A', 'L',
            'qB', 'w', 'rk', 'q',
            'hh_wealth',
            'clearing_Y', 'fisher_res', 'w_res', 'invest_res', 'valuation_res', 'NKPC_res', 'NKPC_w_res',
            'Ip', 'Pi_w_increase', 'Pi_increase',
            'eg', 'em',
            'Z', 's', 's_w', 'psi',
            'p_eq', 'p_share', 'p_k', 'Div_k', 'p_int', 'Div_int']

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = blocks.block_pre
        self.block_post = blocks.block_post

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1  # number of fixed discrete states (here discount factor)
        par.Nz = 7  # number of stochastic discrete states (here productivity)

        # a. preferences
        par.sigma = 2.0  # CRRA coefficient
        par.beta_mean =  0.97950170 # 0.950532**(1/4)  # discount factor, mean, range is [mean-width,mean+width]
            # 0.92045 = weighted average in Auclert et al (2020) Humps, Jumps
        par.beta_delta = 0.00000  # discount factor, width, range is [mean-width,mean+width]
        # par.beta = 0.92045
        # par.beta_solver_init = 0.9505


        # b. income parameters
        par.rho_e = 0.966  # AR(1) parameter
        par.sigma_e = 0.50  # std. of e
        par.Ne = par.Nz

        # c. price setting
        # relevant markup parameters (epsilon_p, v_p, xi_w, epsilon_w, v_w)
        par.mu_p = 1.06
        par.e_p = par.mu_p/(par.mu_p-1)
        par.e_w = par.e_p
        par.xi_p = 0.926    # calvo price stickiness
        par.xi_w = 0.899    # calvo wage stickiness
        par.v_p = 0        # Kimball superelasticity for prices
        par.v_w = 0        # Kimball superelasticity for wages
        # par.kappa_p = 0.1   # slope of Phillips curve
        par.phi_K = 9.0      # elasticity of investment

        # d. government
        par.rho_m = 0.6  # Taylor rule intertia - 0.89
        par.phi_pi = 1.5  # Taylor rule coefficient
        par.phi_tau = 0.1     # Response of tax rate to debt (p.a.)
        par.phi_G = 0     # Tax financing of government expenditure shock

        # spending targets
        # bond supply targets

        # e. calibration
        # set targets
        r_ss_target_p_a = 0.05
        xi_p_a = 0.065  # Intermedation spread (p.a.)
        par.r_ss_target = (1 + r_ss_target_p_a)**(1/4) - 1
        par.xi =  1 + par.r_ss_target - (1+ r_ss_target_p_a - xi_p_a)**(1/4)
        par.frisch = 0.5    #  Frisch elasticity
        par.alpha = np.nan   #  Capital share - calibrated to match K = K_target
        par.delta_K = 0.053 #  Depreciation of capital (p.a.)
        par.K_Y_ratio = 2.23    #  Capital to GDP (p.a.)
        par.L_Y_ratio = 0.23    #  Liquid assets to GDP (p.a.)
        par.G_Y_ratio = 0.16    # Spending-to-GDP
        par.qB_Y_ratio = 0.46   # Government bonds to GDP (p.a.)
        par.maturity = 5*4    #  Maturity of government debt (a.)
        par.delta_q = (par.maturity - 1) * (1 + par.r_ss_target) / par.maturity
        par.chi = 0.005*4  # redistribution share for illiquid assets
        par.hh_wealth_Y_ratio = np.nan   # Total household wealth to GDP - assinged to steady state value
        par.nu = np.nan  # scaling factor in disutility from labor
        par.Theta = np.nan # productivity factor
        par.A_target = np.nan  # illiquid asset target (needs to be adapted in the case of different groups)

        # sticky information parameter
        par.inattention = 0. # 0.935


        # e. grids
        # TODO: Think about if these grids are enough and max values fit
            # especially for the liquid assets it might not be enough
        par.l_min = 0.0  # maximum point in grid for a
        par.l_max = 10.0  # maximum point in grid for a
        par.Nl = 50  # number of grid points

        par.a_min = 0.0  # maximum point in grid for a
        par.a_max = 10.0  # maximum point in grid for a
        par.Na = 50  # number of grid points

        # f. shocks
        par.jump_eg = 0.01 # 0.01  # initial jump
        par.rho_eg = 0.6 # AR(1) coefficient
        par.std_eg = 0.  # std. of innovation
        par.jump_em = 0.  # initial jump
        par.rho_em = 0.6 # AR(1) coefficient
        par.std_em = 0.  # std. of innovation

        # h. misc.
        par.T = 200  # length of transition path
        par.simT = 100  # length of simulation

        par.max_iter_solve = 50_000  # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000  # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100  # maximum number of iteration when solving eq. system

        par.tol_ss = 1e-11  # tolerance when finding steady state
        par.tol_solve = 1e-12  # tolerance when solving household problem
        par.tol_simulate = 1e-12  # tolerance when simulating household problem
        par.tol_broyden = 1e-10  # tolerance when solving eq. system

        par.start_dbeg_opti = True

    def allocate(self):
        """ allocate model """

        par = self.par

        par.beta_grid = np.zeros(par.Nfix)

        self.allocate_GE()

        # add additional dimension to policy indices and weights
        if len(self.pols_hh) == 2:
            assert self.pols_hh == self.grids_hh, 'Histogram method only works if grids is over both policies'
            sol_shape = (2, *self.ss.pol_indices.shape)
            path_pol_shape = list(self.path.pol_indices.shape)
            path_pol_shape.insert(1, 2)
            path_pol_shape = tuple(path_pol_shape)
            sim_pol_shape = list(self.sim.pol_indices.shape)
            sim_pol_shape.insert(1, 2)
            sim_pol_shape = tuple(sim_pol_shape)

            self.ss.pol_indices = np.zeros(sol_shape, dtype=np.int_)
            self.ss.pol_weights = np.zeros(sol_shape)
            self.path.pol_indices = np.zeros(path_pol_shape, dtype=np.int_)
            self.path.pol_weights = np.zeros(path_pol_shape)
            self.sim.pol_indices = np.zeros(sim_pol_shape, dtype=np.int_)
            self.sim.pol_weights = np.zeros(sim_pol_shape)



    # override steady state functions
    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

    # override helper functions
    _find_i_and_w_dict = replaced_functions._find_i_and_w_dict
    _find_i_and_w_path = replaced_functions._find_i_and_w_path


