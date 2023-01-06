""" Collection of functions that generate the analysis for the thesis in a child class of the GEModelClass."""


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

    def calc_FMP(self):
        """Calculate fiscal multiplier"""

        ss = self.ss
        par = self.par
        path = self.path

        assert path.eg.any(), 'No fiscal policy shocks specified'
        assert ss.Y != 0.0, 'ss.Y == 0 -> divide error in fiscal multiplier'
        assert ss.G != 0.0, 'ss.G == 0 -> divide error in fiscal multiplier'

        fmp_impact = (path.Y[0,0] - ss.Y) / (path.G[0,0] - ss.G)

        dY = np.array([(1 + ss.r) ** (-t) * (path.Y[0, t] - ss.Y) for t in range(par.T)])
        dG = np.array([(1 + ss.r) ** (-t) * (path.G[0, t] - ss.G) for t in range(par.T)])

        fmp_cum = dY.sum() / dG.sum()

        return fmp_impact, fmp_cum
