""" Collection of functions that generate the analysis for the thesis in a child class of the GEModelClass."""


import numpy as np
import datetime
import numpy as np


from GEModelTools.GEModelClass import GEModelClass


class HANKStickyAnalyticsClass(GEModelClass):

    def calc_MPC(self, annual=True, income='labor'):
        """ calculate MPCs
            :param annual: calculate annual MPC. """

        ss = self.ss
        par = self.par
        jac_hh = self.jac_hh

        if income == 'labor':
            # MPC from labor income
            assert jac_hh['C_hh', 'Z'].any(), 'Household Jacobian not calculated yet'

            if annual:
                mpc = sum([(1/(1+ss.r))**t * jac_hh['C_hh', 'Z'][t, 0] for t in range(4)])
                # print(f'aggregate annual MPC out of labor income: {mpc:.3f}')
            else:
                mpc = jac_hh['C_hh', 'Z'][0, 0]
                # print(f'aggregate MPC out of labor income: {mpc:.3f}')
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
            # print(f'aggregate  MPC: {mpc:.3f} [annual: {annual}]')

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
            # print(f'aggregate  MPC: {mpc:.3f} [annual: {annual}]')
        else:
            raise NotImplementedError

        return mpc

    def calc_FMP(self, cum_FMP_max_T=None):
        """Calculate fiscal multiplier"""

        ss = self.ss
        par = self.par
        path = self.path

        if cum_FMP_max_T == None:
            cum_FMP_max_T = self.par.T

        assert path.eg.any(), 'No fiscal policy shocks specified'
        assert ss.Y != 0.0, 'ss.Y == 0 -> divide error in fiscal multiplier'
        assert ss.G != 0.0, 'ss.G == 0 -> divide error in fiscal multiplier'

        fmp_impact = (path.Y[0,0] - ss.Y) / (path.G[0,0] - ss.G)

        dY = np.array([(1 + ss.r) ** (-t) * (path.Y[0, t] - ss.Y) for t in range(par.T)])
        dG = np.array([(1 + ss.r) ** (-t) * (path.G[0, t] - ss.G) for t in range(par.T)])

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

        if baseline_model == None:
            print("\r" + 'Find steady state  ', end="")
            model.find_ss(do_print=False)
        else:
            print("\r" + f'Use steady state from {baseline_model.name}')
            model = baseline_model.copy()
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
    def get_sticky_IRFs(model, inattention=0.935):
        """ get sticky information IRFs for list of models"""

        m_sticky = f"{model.name}_sticky"

        print('\r' + f' -------- Model: {m_sticky} ---------')

        model_sticky = model.copy()

        model_sticky.name = m_sticky

        model_sticky.par.inattention = inattention

        print("\r" + 'Compute Jacobians', end="")
        model_sticky.compute_jacs(do_print=False)
        print("\r" + 'Find IRFs', end="")
        model_sticky.find_IRFs(do_print=False)

        print("\r" + '                  ', end="")

        return model_sticky