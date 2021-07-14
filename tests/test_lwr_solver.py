# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np
from lwr.lwr_solver import LWR_Solver
from lwr import config
from LWR_posterior import BC_prior_mean_OU_highres, logPriorLWR, logLikelihoodLWR


class TestLWRSolver(unittest.TestCase):
    """
    Tests for the LWR_Solver class
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_posterior(self):
        le_outlet = np.log(np.genfromtxt("data/MCMC_ICs/outlet_2K.txt")) - BC_prior_mean_OU_highres['BC_outlet']
        le_inlet = np.log(np.genfromtxt("data/MCMC_ICs/inlet_2K.txt")) - BC_prior_mean_OU_highres['BC_inlet']
        # MCMC_1, process0, iloc[2000]
        FD = {'z': 175.64524045695617,
         'rho_j': 379.3928422197564,
         'u': 3.99525422856635,
         'w': 0.19811128675617157,
          "logBC_outlet_centred": le_outlet,
        "logBC_inlet_centred": le_inlet
             }
        lelogpost = logPriorLWR(**FD) + logLikelihoodLWR(**FD)
        assert round(lelogpost, 2) == -3269.51
