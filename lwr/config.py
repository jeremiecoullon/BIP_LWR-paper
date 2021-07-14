# -*- coding: utf-8 -*-

import os
import numpy as np


BASE_PATH = os.path.abspath(os.path.dirname(__file__))
# binary files
BIN_PATH = os.path.join(BASE_PATH, 'bin')
DATA_PATH = os.path.join(BASE_PATH, 'traffic_data')

default_config_dict = {'my_analysis_dir': '2018/Default_directory',
                    'run_num': 1,
                    'data_array_dict': {'flow': 'test_data/artificial_data_array_flow_prior_BC_poisson_2.csv',
                                    'density': 'test_data/artificial_data_array_density_prior_BC.csv'},
                      }


class ConfigHolder:
    """
    Class to hold configuration variables for a generic MCMC sampler
    """
    def __init__(self, save_chain=False):
        self.save_chain = save_chain


class LWRConfigHolder(ConfigHolder):
    """
    Class to hold configuration variables for a LWR mcmc sampler
    """
    def __init__(self, my_analysis_dir, run_num, data_array_dict,
                upload_to_S3=False, save_chain=True, comments="",
                root_analysis_folder="../", ratio_times_BCs=40, step_save=10, FD_only=False, **kwargs):
        super(LWRConfigHolder, self).__init__(save_chain=save_chain)

        self.my_analysis_dir = my_analysis_dir
        self.RUN_NUM = run_num
        self.data_array_dict = data_array_dict
        self.upload_to_S3 = upload_to_S3
        self.comments = comments
        self.ratio_times_BCs = ratio_times_BCs
        self.step_save = step_save
        self.FD_only = FD_only

        # ========================
        self.DATA_VARIABLE = 'flow'
        self.ERROR_MODEL = 'poisson'
        self.root_analysis_folder = root_analysis_folder
        self.DATA_TRANSFORM = lambda x:x
        self.BC_dict = {'flow': '', 'density': 'rho_adaptive_70108_longer.csv'}

        self.setup_vars()

    def setup_vars(self):
        """
        Setup various paths based on configuration variables
        """
        self.BASE_PATH = os.path.abspath(os.path.dirname(__file__))
        # BASE_PATH = "/Users/jeremiecoullon/documents/phd/scripts/interactive/BIP_LWR"
        # path to data used to run BIP_LWR estimation scripts
        self.DATA_PATH = os.path.join(self.BASE_PATH, 'traffic_data')
        # binary files
        self.BIN_PATH = os.path.join(self.BASE_PATH, 'bin')

        # current outputs of analysis
        self.ANALYSIS_DIR = os.path.join(os.path.abspath('{}Analysis'.format(self.root_analysis_folder)), self.my_analysis_dir)
        # MCMC outputs
        self.MCMC_OUTPUT = os.path.join(self.ANALYSIS_DIR, 'MCMC_outputs')
        # ----------------------------------------------------
        # ----------------------------------------------------


        # data paths
        self.DATA_ARRAY = os.path.join(self.DATA_PATH, self.data_array_dict.get(self.DATA_VARIABLE))

        self.DATA_FLOW_PATH = os.path.join(self.DATA_PATH, self.data_array_dict.get('flow'))
        self.DATA_DENSITY_PATH = os.path.join(self.DATA_PATH, self.data_array_dict.get('density'))
