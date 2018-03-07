import numpy as np

from scipy.linalg import block_diag
from gpode.linlatentforcemodels import (linalg_util,
                                        matrixrv_util)
from collections import namedtuple
from gpode.kernels import GradientMultioutputKernel


Dimensions = namedtuple("Dimensions",
                        "R K N")

class varmlfm_ns:
    def __init__(self,
                 model_mats,
                 data_times,
                 data_Y,
                 sigmas,
                 gammas,
                 g_gp_pars)
        pass

    _handle_data(self, data_times, data_Y)

