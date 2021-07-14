# -*- coding: utf-8 -*-

import numpy as np

def remove_IC_influence(data_array_str):
    """
    Returns minimum value of time to compute the likelihood based on the data used.
    This removes the influence of the boundary conditions on the likelihood for reasonable ranges of wavespeeds


    Parameters
    ----------
    data_array_str: str
        Name of data_array (CSV file) for flow data

    Returns
    -------
    t_min: int
        Minimum value of time to compute the likelihood for
    """
    data_IC_dict = {'data_array_70108_flow_shorter2.csv': 376, # DS1,60t IC stops influencing log-lik at 374
            'test_data/Simulated_LWR_Nov2018/data_array_DelCast_flow.csv': 376, # DelCast Simluated data DS1,60t IC
            'test_data/Simulated_LWR_Nov2018/data_array_DelCast_flow_RT_40.csv': 376, # DelCast Simluated data; high dimensional BCs DS1,60t IC
            'test_data/Simulated_LWR_Nov2018/data_array_Exp_flow.csv': 376, # Exp FD Simluated data DS1,60t IC
            'test_data/Simulated_LWR_Nov2018/data_array_Exp_flow_RT40.csv': 376, # Exp FD Simluated data DS1,60t IC
            'Sim_data_array_Exp_flow_40t.csv': 387, # Exp Simulated, 40t
            'Sim_data_array_DelCast_flow_40t.csv': 387, # Exp Simulated, 40t

            'data_array_70108_flow_52t.csv': 384, # DS1, 52t. Cut 6 times. IC stops influencing at 383 for FD_cutIC_top_mode, and 382 for FD_cutIC_bottom_mode
            'data_array_70108_flow_49t.csv': 387, # DS1, 49t. Cut 6 times. IC stops influencing at 386 for FD_cutIC_top_mode, and 385 for FD_cutIC_bottom_mode
            'data_array_70108_flow_48t.csv': 388, # DS1, 48t. Cut 6 times.
            'data_array_70108_flow_47t.csv': 389, # DS1, 47t. Cut 6 times.
            'data_array_70108_flow_46t.csv': 390, # DS1, 46t. Cut 6 times. IC stops influencing at 388
            'data_array_70108_flow_60t_halftime.csv': 374, #DS1, 60t, half-space. IC stops influencing at 372.
            'data_array_flow_70108_longer.csv': 356, # DS1, 150t: IC stops influencing log-lik at 352 for Exp_FD, and 354 for del_Cast
            'test_data/simulated_data_Oct2/data_array_Simluated2_priorBC_Oct2018_flow.csv': 376, # Simulated data with BC from prior (60t). Influence stops at t>=384
            'test_data/artificial_data_array_flow_prior_BC_poisson_short.csv': 350, # don't remove IC influence for running tests
            }
    if data_array_str in data_IC_dict.keys():
        t_min = data_IC_dict[data_array_str]
    else:
        raise ValueError("IC influence wasn't removed from the log-likelihood for this dataset")
    return t_min

def create_lwr_grid(data_array, BC_len):
    """
    Creates mapping between LWR grid and data detectors.
    Also calculates the number of PDE cells needed so that data detectors hit the
    center of cells.

    Parameters
    ---------
    data_array: ndarray
        Array of data in format ('x', 't', 'flow')
    BC_len: int
        Length of BCs

    Returns
    -------
    data_to_PDE_time: dict
        Mapping between data time indexes and PDE output indexes
    data_to_PDE_space: dict
        Mapping for space
    PDE_num_cells: int
        Number of cells for the PDE solver
    """
    data_spaces = np.unique(data_array[:, 0])
    data_times = np.unique(data_array[:, 1])
    final_time = int(data_times[-1] - data_times[0])
    # don't include first and last detectors in data_spaces, as they correspond to ghost cells
    data_full_res = np.arange(0.5, data_spaces[-1],0.5)
    num_cells_data = len(data_full_res)
    num_intermediate_cells = 25 # choosing this to be 25 means we get 259 PDE cells
    # To get the PDE cells centers to be at the data points, we need:
    # num_cells_data: number of detectors (assuming no missing detectors!).
    # Note: num_cells_data doesn't include the ghost cells (namely the first and last ones)
    PDE_num_cells = (num_cells_data) +  num_intermediate_cells * (num_cells_data + 1)

    # we've already chosen PDE_num_cells such that data centers are hit: PDE_num_cells
    # Now create list of indexes (in space) for the PDE output that correspond to detectors (assuming no missing detectors)
    PDE_data_full = list(range(PDE_num_cells))[ num_intermediate_cells::num_intermediate_cells+1]

    # index of data_full_res that match detectors we actually have
    data_idx = [i[0] for i in enumerate([elem in data_spaces for elem in data_full_res]) if i[1] == True]

    # indexes to keep in PDE output (correspond to detectors we have)
    PDE_keep_space = [PDE_data_full[elem] for elem in data_idx]

    # define a dictionary that maps detector number to PDE index.
    data_to_PDE_space = {elem[0]:elem[1] for elem in zip(data_spaces[1:-1], PDE_keep_space)}
    # Same for time
    # range of times: from 0 until final_time+1. Note that the size of rho_claw should not depend on the resolution of the BCs
    data_to_PDE_time = {elem[0]:elem[1] for elem in zip(data_times, list(range(0, final_time+1)))}

    return data_to_PDE_time, data_to_PDE_space, PDE_num_cells
