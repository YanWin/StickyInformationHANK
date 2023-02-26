""" Collection of functions that generate the analysis for the thesis in a child class of the GEModelClass."""


import numpy as np
import datetime
import numpy as np


from GEModelTools.GEModelClass import GEModelClass


class HANKStickyAnalyticsClass(GEModelClass):

    def check_non_lin(self, p):

        par = self.par
        ss = self.ss
        path = self.path

        non_lin = False

        for t in range(par.T):
            D_agg = path.D[0].sum(axis=(0,1,2))     # aggregate mass on illiquid asset grid
            i_a_pos = np.argwhere(D_agg > 1e-10)    # find indices with positive mass on illiquid asset grid
            for i_a in i_a_pos:
                lowest_income = (1 + path.rl[p,t]) * par.l_grid[0] + path.Z[p,t] * par.z_grid[0]
                highest_redistribution = ss.ra / (1 + ss.ra) * (1 + path.ra[p,t]) * par.a_grid[i_a] + par.chi * (
                        (1 + path.ra[p,t]) * par.a_grid[i_a] - (1 + ss.ra) * par.A_target)
                if lowest_income + highest_redistribution < 0:
                    non_lin = True
        if non_lin:
            print("negative cash-on-hand possible given paths"
                  "-> non-linearities in policy functions")
        else:
            print("no non-linearities in the policy functions")

    def calc_MPC(self, annual=True, income='labor'):
        """ calculate MPCs
            :param annual: calculate annual MPC. """

        ss = self.ss
        par = self.par
        jac_hh = self.jac_hh

        if income in ['labor', 'tax']:
            if income == 'labor':
                inputname = 'wN'
            elif income == 'tax':
                inputname = 'tau'

            # MPC from labor income
            assert jac_hh[('C_hh', inputname)].any(), 'Household Jacobian not calculated yet'

            if annual:
                mpc = sum([(1/(1+ss.r))**t * jac_hh[('C_hh', inputname)][t, 0] for t in range(4)])
                # print(f'aggregate annual MPC out of labor income: {mpc:.3f}')
            else:
                mpc = jac_hh[('C_hh', inputname)][0, 0]
        elif income == 'liquid':
            # for liquid assets
            MPC = np.zeros(ss.D.shape)
            dc = (ss.c[:, :, 1:, :] - ss.c[:, :, :-1, :])
            l_grid_full = np.repeat(par.l_grid, par.Na).reshape(1, 1, par.Nl, par.Na)
            dl = (1 + ss.rl) * l_grid_full[:, :, 1:, :] - (1 + ss.rl) * l_grid_full[:, :, :-1, :]
            MPC[:, :, :-1, :] = dc / dl
            MPC[:, :, -1, :] = MPC[:, :, -2, :]  # assuming constant MPC at end
            mpc = np.sum(MPC * ss.D)
            if annual:
                mpc = 1 - (1 - mpc) ** 4
            mpc = 1 - (1 - mpc) ** 4

        elif income == 'illiquid':
            # for illiquid assets
            MPC = np.zeros(ss.D.shape)
            dc = (ss.c[:, :, :, 1:] - ss.c[:, :, :, :-1])
            a_grid_full = np.repeat(par.a_grid, par.Nl).reshape(1, 1, par.Nl, par.Na).swapaxes(2, 3)
            da = (1 + ss.ra) * a_grid_full[:, :, :, 1:] - (1 + ss.ra) * a_grid_full[:, :, :, :-1]
            MPC[:, :, :, :-1] = dc / da
            MPC[:, :, :, -1] = MPC[:, :, :, -2]  # assuming constant MPC at end
            mpc = np.sum(MPC * ss.D)
            if annual:
                mpc = 1 - (1 - mpc) ** 4
            mpc = 1 - (1 - mpc) ** 4
        else:
            raise NotImplementedError

        return mpc

    def calc_FMP(self, cum_FMP_max_T=None):
        """Calculate fiscal multiplier"""

        ss = self.ss
        par = self.par
        IRF = self.IRF

        if cum_FMP_max_T == None:
            cum_FMP_max_T = self.par.T

        assert IRF['eg'].any(), 'No fiscal policy shocks specified'
        assert IRF['G'][0] != 0.0, 't=0, G == 0 -> divide error in fiscal multiplier'

        fmp_impact = IRF['Y'][0] / IRF['G'][0]

        dY = np.array([(1 + ss.r) ** (-t) * IRF['Y'][t] for t in range(par.T)])
        dG = np.array([(1 + ss.r) ** (-t) * IRF['G'][t] for t in range(par.T)])

        fmp_cum = dY[:cum_FMP_max_T].sum() / dG[:cum_FMP_max_T].sum()

        return fmp_impact, fmp_cum

    def plot_jacs(self):
        """ adaption of the test_jacs function with more options"""
        pass

    @staticmethod
    def solution_routine(model, baseline_model=None, update_par={}, do_non_linear=False, save_model=False):
        """ solution routine to solve a model

            :param model: initialied model.
            :param baseline_model: use steady state from this baseline model and only compute jacs and IRFs.
            :param update_par: specify parameters to update from baseline model.
            :param do_non_linear: find non-linear transition path if TRUE.
            :param save_model: save model with date timestamp if TRUE.

            :return returns the model"""

        name = model.name

        if baseline_model == None:
            print("\r" + 'Find steady state  ', end="")
            model.find_ss(do_print=False)
        else:
            print("\r" + f'Use steady state from {baseline_model.name}')
            model = baseline_model.copy()
            model.name = name
            for p in update_par.keys():
                model.par.__dict__[p] = update_par[p]

        print("\r" + 'Compute Jacobians  ', end="")
        model.compute_jacs(do_print=False)
        if do_non_linear:
            print("\r" + 'Find transition path', end="")
            model.find_transition_path(do_print=False, do_print_unknowns=False)
        print("\r" + 'Find IRFs           ', end="")
        model.find_IRFs(do_print=False)

        if save_model:
            m_name = model + f'_{datetime.datetime.now().strftime("%m_%d")}'
            model.name = m_name
            model.save()
            print("\r" + f'saved {m_name}')

        return model

    @staticmethod
    def get_sticky_IRFs(model, inattention=0.935, name=None):
        """ get sticky information IRFs for model"""

        model_sticky = model.copy()

        if name == None:
            m_sticky = f"{model.name}_sticky"
        else:
            m_sticky = name

        print('\r' + f' -------- Model: {m_sticky} ---------')

        model_sticky.name = m_sticky

        model_sticky.par.inattention = inattention

        print("\r" + 'Compute Jacobians', end="")
        model_sticky.compute_jacs(do_print=False)
        print("\r" + 'Find IRFs', end="")
        model_sticky.find_IRFs(do_print=False)

        print("\r" + '                  ', end="")

        return model_sticky