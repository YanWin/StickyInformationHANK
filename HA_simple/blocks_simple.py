import numpy as np
import numba as nb

from GEModelTools import lag, lead


@nb.njit
def block_pre(par, ini, ss, path, ncols=1):
    """ evaluate transition path - before household block """
    pass


@nb.njit
def block_post(par, ini, ss, path, ncols=1):
    """ evaluate transition path - after household block """
    pass