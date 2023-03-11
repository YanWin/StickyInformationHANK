import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass
from GEModelTools import tests

from HA_simple import steady_state_simple
from HA_simple import household_problem_simple
from HA_simple import blocks_simple


class HAsimpleModelClass(EconModelClass, GEModelClass):
    def __init__(self, savefolder='saved', *args, **kwargs):
        EconModelClass.__init__(self, *args, **kwargs)
        self.savefolder = savefolder


    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','sim','ss','path']

        # b. household
        self.grids_hh = ['l']  # grids
        self.pols_hh = ['l']  # policy functions
        self.inputs_hh = ['y','rl','ey']  # direct inputs
        self.inputs_hh_z = []  # transition matrix inputs
        self.outputs_hh = ['c','l']  # outputs
        self.intertemps_hh = ['vbeg_l']  # intertemporal variables

        # c. GE
        self.shocks = ['ey']  # exogenous shocks
        self.unknowns = []  # endogenous unknowns
        self.targets = ['MPC_match']  # targets = 0
        self.blocks = [
            'hh'
        ]

        # d. all other variables
        self.varlist = [
            'MPC_match',
            'ey',
            'y',
            'rl',
            'r'
            ]

        # e. functions
        self.solve_hh_backwards = household_problem_simple.solve_hh_backwards


    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1  # number of fixed discrete states (here discount factor)
        par.Nz = 7  # number of stochastic discrete states (here productivity)

        # targets
        r_ss_target_p_a = 0.05
        par.r_ss_target = (1 + r_ss_target_p_a)**(1/4) - 1  # quarterly

        xi_p_a = 0.065  # intermediation spread (p.a.)
        par.xi = 1 + par.r_ss_target - (1 + r_ss_target_p_a - xi_p_a) ** (1 / 4)  # quarterly

        d_a = par.r_ss_target * 14.36 # steady state distribution from illiquid asset
        par.Z_target = 0.53 + d_a
        # MPCs_data = [0.525, 0.175, 0.10, 0.045, 0.03, 0.025]
        par.MPC_target = 0.525

        assert par.Nfix == 1

        # a. preferences
        par.sigma = 1.0  # CRRA coefficient
        # TODO: change to new quarterly beta
        par.beta_mean = 0.9951   # discount factor, mean, range is [mean-width,mean+width]
        par.beta_delta = 0.00000  # discount factor, width, range is [mean-width,mean+width]
        par.inattention = 0.0

        # b. income and saving parameters
        par.rho_e = 0.966  # AR(1) parameter
        par.sigma_e = 0.5  # 0.5 std. of e


        # i. grids
        par.l_min = 0.0  # maximum point in grid for a
        par.l_max = 10.0*3  # maximum point in grid for a  - quarterly
        par.Nl = 100  # number of grid points

        # j. shocks
        # 3. persistent income shock
        par.jump_ey = 0.0  # initial jump
        par.rho_ey = 0.0  # AR(1) coefficient
        par.std_ey = 0.0  # std. of innovation

        # k. misc.
        par.T = 250  # length of transition path
        par.simT = 100  # length of simulation

        par.max_iter_solve = 50_000  # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000  # maximum number of iterations when simulating household problem
        par.max_iter_broyden = 100  # maximum number of iteration when solving eq. system

        par.tol_ss = 1e-12  # tolerance when finding steady state
        par.tol_solve = 1e-12  # tolerance when solving household problem
        par.tol_simulate = 1e-12  # tolerance when simulating household problem
        par.tol_broyden = 1e-12  # tolerance when solving eq. system

        par.py_hh = False # call solve_hh_backwards in Python-model
        par.py_block = False # call blocks in Python-model
        par.full_z_trans = False # let z_trans vary over endogenous states


    def allocate(self):
        """ allocate model """

        par = self.par
        par.beta_grid = np.zeros(par.Nfix)

        self.allocate_GE(update_varlist=False)


    # override steady state functions
    prepare_hh_ss = steady_state_simple.prepare_hh_ss
    find_ss = steady_state_simple.find_ss

