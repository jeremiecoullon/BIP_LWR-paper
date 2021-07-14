# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
from lwr.config import LWRConfigHolder
from lwr.config import BIN_PATH, default_config_dict
import sys
sys.path.append(BIN_PATH) # path to bin directory for Fortran riemann solvers
import lwr_del_Cast
from clawpack import pyclaw
import math
import os
from .solver_util import create_lwr_grid, remove_IC_influence


def load_CSV_data(path):
    """
    Load CSV from the data folder (`traffic_data`)

    Parameters
    ----------
    path: str
        Path relative to `traffic_data` folder

    Returns
    -------
    data: ndarray
        Array of data
    """
    return np.genfromtxt(os.path.join('data/traffic_data', path))

class LWR_Solver:
    """
    Class for LWR solver (for different FDs) with variout helper function

    Parameters
    ----------
    data_array : ndarray or None
        array of raw data (3 columns) to use to compute the loss
        If None: use data_array in config.py
    final_time: int
        End time for PDE solver. If this is smaller than the final time in data_array,
        then cut data_array
    config_dict: dict
        Dictionary of configuration parameters
    """
    def __init__(self, data_array=None, final_time=None, config_dict=default_config_dict):

        self.config = LWRConfigHolder(**config_dict)
        if data_array is None:
            self.data_array = np.genfromtxt(self.config.DATA_ARRAY)
        else:
            self.data_array = data_array

        # get unique time and space coordinates for the raw data
        self.data_spaces = np.unique(self.data_array[:, 0])
        self.data_times = np.unique(self.data_array[:, 1])

        # final_time is in minutes
        if final_time is None:
            self.final_time = int(self.data_times[-1] - self.data_times[0])
        else:
            raise ValueError("don't define final_time manually")
            data_array_final_time = int(self.data_times[-1] - self.data_times[0] + 1)
            if final_time > data_array_final_time:
                raise ValueError("'final_time' argument should be less than {} (number of minutes in data)".format(data_array_final_time))
            self.final_time = final_time
            # keep subset of data_array
            t_max = self.data_times[0] + self.final_time
            self.data_array = self.data_array[(self.data_array[:, 1] >= self.data_spaces[0]) & (self.data_array[:, 1] < t_max)]

        self.x_min_solver = int(self.data_spaces[0])
        self.x_max_solver = int(self.data_spaces[-1])

        # Note: self.ratio_times should now be 1
        # this should be a multiple of the number of minutes in data (ie: len(data_times) )
        self.ratio_times = 1
        self.BC_len = self.config.ratio_times_BCs * int(self.final_time) + 1 # Number of PDE outputs should be the same as BC length
        self.out_times = np.linspace(0, self.final_time*self.config.ratio_times_BCs, self.final_time+1, endpoint=True) # array of desired output times to pass to Clawpack

        data_to_PDE_time, data_to_PDE_space, PDE_num_cells = create_lwr_grid(data_array=self.data_array, BC_len=self.BC_len)

        self.data_to_PDE_time = data_to_PDE_time
        self.data_to_PDE_space = data_to_PDE_space
        self.PDE_num_cells = PDE_num_cells

        # data without the BCs (ie: rows with the first and last detector)
        self.df_data = pd.DataFrame(self.data_array[(self.data_array[:, 0]!=self.data_spaces[0]) & (self.data_array[:, 0]!=self.data_spaces[-1])],
                        columns = ['space','time', 'data'])


        # Create BCs with resolution `final_time`
        self.density_data_array = np.genfromtxt(self.config.DATA_DENSITY_PATH)
        first_space = self.data_spaces[0]
        last_space = self.data_spaces[-1]
        self.BC_outlet = self.density_data_array[self.density_data_array[:, 0]==last_space][:,2]
        self.BC_inlet = self.density_data_array[self.density_data_array[:, 0]==first_space][:,2]

        min_time = min(self.density_data_array[:, 1])
        IC_raw = self.density_data_array[self.density_data_array[:, 1]==min_time][:,2]
        # interpolate raw BCs
        # note that the raw data isn't evenly spaced. So you need to use data_spaces in the interpolation
        self.data_IC = np.interp(np.arange(0, 1 , 1/self.PDE_num_cells), self.data_spaces*(1/last_space), IC_raw)

        # get true BCs from data_array. This is used in the likelihood to constain wacky FD (especially when using flow data)
        self.true_BC_inlet = self.data_array[self.data_array[:, 0]==self.data_spaces[0]][:, 2]
        self.true_BC_outlet = self.data_array[self.data_array[:, 0]==self.data_spaces[-1]][:, 2]



    def lwr(self, BC_inlet=None, BC_outlet=None, **kwargs):
        """
        LWR solver
        del_Cast_w, del_Cast_u, del_Cast_Z: parameters in the exponential FD

        Parameters
        ----------
        BC_inlet, BC_outet : BCs to pass to LWR. If none, use BCs from data.
        **kwargs :
            FD paramters to pass to LWR. one of them must be the solver name (ex: 'solver' : 'lwr_exp')
        """
        BC_outlet, BC_inlet = self.process_BCs(BC_outlet, BC_inlet)
        def custom_bc_lower(state, dim, t, qbc, auxbc, num_ghost):
            """
            user defined boundary conditions at the inlet
            """
            qbc[:,:num_ghost] = BC_inlet[int(t)]

        def custom_bc_upper(state, dim, t, qbc, auxbc, num_ghost):
            """
            user defined boundary conditions at the outlet
            """
            qbc[0,-num_ghost:] = BC_outlet[int(t)]

        le_solver = kwargs.get('solver')
        if le_solver == 'lwr_del_Cast':
            riemann_solver = lwr_del_Cast
        elif le_solver == 'lwr_exp':
            riemann_solver = lwr_exp
        else:
            raise ValueError("solver not implemented")
        #===========================================================================
        # Set up solver and solver parameters
        #===========================================================================

        solver = pyclaw.ClawSolver1D(riemann_solver)
        solver.kernel_language = 'Fortran'

        solver.num_eqn = 1
        solver.num_waves = 1

    #     custom boundary conditions
        solver.bc_lower[0] = pyclaw.BC.custom
        solver.bc_upper[0] = pyclaw.BC.custom
        solver.user_bc_lower = custom_bc_lower
        solver.user_bc_upper = custom_bc_upper


        #======================================
        # Set up domain and initialize solution
        #======================================
        x = pyclaw.Dimension(self.x_min_solver, self.x_max_solver, self.PDE_num_cells, name='x')         # spatial grid
        domain = pyclaw.Domain(x)
        num_eqn = 1
        state = pyclaw.State(domain,solver.num_eqn)

        grid = state.grid
        xc=grid.x.centers

        # Stopped traffic at a light:
        # constant_density = 30
        # state.q[0,:] = constant_density*(xc<=0) + constant_density*(xc>0.)

        # Initial conditions from data
        state.q[0,:] = self.data_IC

        state.problem_data['efix']=True

        if 'z' in kwargs:
            kwargs['z'] = kwargs.get('z')/self.config.ratio_times_BCs
        if 'alpha' in kwargs:
            kwargs['alpha'] = kwargs.get('alpha')/self.config.ratio_times_BCs
        for key,value in list(kwargs.items()):
            if key!='solver':
                state.problem_data['{}'.format(key)] = value

        #===========================================================================
        # Setup controller and controller parameters. Then solve the problem
        #===========================================================================
        # HACK: remove log file
        try:
            os.remove('pyclaw.log')
        except OSError:
            pass
        claw = pyclaw.Controller()
        claw.verbosity = 2
        # HACK: time is 220 - 1 because of the BC fix
        claw.tfinal = self.final_time*self.config.ratio_times_BCs
        claw.solution = pyclaw.Solution(state,domain)
        claw.solver = solver
        # claw.num_output_times = self.final_time                        # number of frames printed
        claw.output_style = 2
        claw.out_times = self.out_times
        claw.keep_copy = True
        claw.output_format = None


        status = claw.run()
        return claw

    def process_BCs(self, BC_outlet, BC_inlet):
        """
        If arguments are `None`, return BC from data.
        Else, make sure the arrays have the correct length

        Parameters
        ----------
        LWR_BC : NoneType or ndarray
            BC array to pass to lwr()

        Returns
        -------
        LWR_BC : ndarray
            BC to pass to lwr()
        """
        if BC_outlet is None:
            BC_outlet = self.BC_outlet
        else:
            if BC_outlet.shape[0] != self.BC_len:
                raise ValueError("Boundary conditions have the wrong shape. They have size {}. Expected size {}".format(len(BC_outlet),
                                                                                    self.BC_len))

        if BC_inlet is None:
            BC_inlet = self.BC_inlet
        else:
            if BC_inlet.shape[0] != self.BC_len:
                raise ValueError("Boundary conditions have the wrong shape. They have size {}. Expected size {}".format(len(BC_inlet),
                                                                                    self.BC_len))

        return BC_outlet, BC_inlet

    def solve_lwr(self, BC_outlet=None, BC_inlet=None, **kwargs):
        """
        - Solve LWR
        - synch up LWR output with data
        - return `PDE_data` and `midas_data`
        """
        claw = self.lwr(BC_outlet=BC_outlet, BC_inlet=BC_inlet , **kwargs)
        # get PDE output in rho_claw
        rho_claw = [elem.q[0,:] for elem in claw.frames]

        # EVALUTATE ONLY AT A FEW DETECTORS
        # detectors: {1.0: 51, 2.0: 103, 2.5: 129, 3.0: 155, 4.0: 207, 4.5: 233}
        # get df_data.time from data_to_PDE_time
        # 1 detector, 1 time
        # df_data_1 = self.df_data[(self.df_data.space == 1) & (self.df_data.time == 454)]
        # # 1 detector, many times
        # df_data_1 = self.df_data[(self.df_data.space == 1) & (self.df_data.time > 400) & (self.df_data.time < 460)]

        # ========================
        t_min = remove_IC_influence(data_array_str=self.config.data_array_dict['flow'])
        df_data_rm_IC_influence = self.df_data[(self.df_data.time >= t_min)]
        # keep times that are a multiple of 3
        if self.config.data_array_dict['flow'] in ['Sim_data_array_DelCast_flow_40t.csv']:
            # df_data_rm_IC_influence = df_data_rm_IC_influence.loc[df_data_rm_IC_influence.time%3==0]
            pass
        # df_data_rm_IC_influence = df_data_rm_IC_influence.loc[df_data_rm_IC_influence.time%3==0]
        # if self.config.data_array_dict['flow'] in ['data_array_70108_flow_shorter2.csv', 'data_array_70108_flow_49t.csv', 'data_array_70108_flow_52t.csv']:
            # remove 3rd detector (which has a different FD)
            # df_data_rm_IC_influence = df_data_rm_IC_influence.loc[df_data_rm_IC_influence.space!=3]
            # pass
        self.df_data = df_data_rm_IC_influence
        # ========================

        # for each row in df_data:
        # get PDE_space_idx from PDE_output corresponding to detector number
        # get PDE_time_idx from PDE_output corresponding to time
        # append a new column of PDE_outputs
        PDE_data = []
        for idx,row in self.df_data.iterrows():
            PDE_space_idx = self.data_to_PDE_space.get(row.space, 'no data at that detector value')
            PDE_time_idx = self.data_to_PDE_time.get(row.time, 'no data at that time')
            PDE_data.append(rho_claw[PDE_time_idx][PDE_space_idx])
            if PDE_space_idx == None:
                sys.exit("PDE_space_idx has a nan value: no data at that detector value. The relevant row in df_data is \n{0}".format(row))
            if PDE_time_idx == None:
                sys.exit("PDE_time_idx has a nan value: no data at that time. The relevant row in df_data is \n{0}".format(row))

        PDE_data = np.array(PDE_data)
        midas_data = self.df_data.data.values
        # Append BCs to PDE_data and midas_data to include in the likelihood computation
        # Check whether BCs have the correct shape
        BC_outlet, BC_inlet = self.process_BCs(BC_outlet, BC_inlet)
        # append LWR BCs to PDE_data
        # keep BC times at each minute only
        PDE_data = np.concatenate((PDE_data, BC_inlet[::self.config.ratio_times_BCs], BC_outlet[::self.config.ratio_times_BCs]))
        # PDE_data = np.concatenate((PDE_data, self.true_BC_inlet, self.true_BC_outlet))
        # append true BCs to midas_data
        midas_data = np.concatenate((midas_data, self.true_BC_inlet, self.true_BC_outlet))
        # convert data to flow using FD
        if self.config.DATA_VARIABLE == 'flow':
            # convert density to flow:
            le_solver = kwargs.get('solver')
            if le_solver == 'lwr_del_Cast':
                FD_del_cast = self._call_FD_neg_power(w=kwargs.get('w'), u=kwargs.get('u'), rho_j=kwargs.get('rho_j'), z=kwargs.get('z'))
                PDE_data = list(map(FD_del_cast, PDE_data))
            elif le_solver == 'lwr_exp':
                FD_exp = self._call_FD_exp(alpha=kwargs.get('alpha'), beta=kwargs.get('beta'))
                PDE_data = list(map(FD_exp, PDE_data))
            else:
                print("solver not implemented")
        elif self.config.DATA_VARIABLE == 'density':
            pass
        return PDE_data,midas_data

    def loss_function(self, BC_outlet=None, BC_inlet=None, loss_sd=0.04, nu=1, **kwargs):
        """
        Calculates squared error loss for LWR for a generic FD. If you pass BCs (as an array),
        then they will be used in the solver. A complexity penalty term will also be used in
        the squared error loss.

        Parameters
        ----------
        **kwargs :
            FD paramters to pass to LWR. one of them must be the solver name (ex: 'solver' : 'lwr_exp')

        Returns
        -------
        LWR_error_square : float
            Squared error for LWR vs traffic data
        """
        PDE_data, midas_data = self.solve_lwr(BC_outlet=BC_outlet, BC_inlet=BC_inlet, **kwargs)
        # Reminder: df_data doesn't include the ghost cells
        if self.config.ERROR_MODEL == 'squared_error':
            PDE_data, midas_data = self.transform_data(PDE_data=PDE_data, midas_data=midas_data, fun_transform=self.config.DATA_TRANSFORM)
            PDE_error = self.squared_error(PDE_data, midas_data, loss_sd)
        elif self.config.ERROR_MODEL == 'poisson':
            PDE_error = self.poisson_error(PDE_data, midas_data)
        else:
            raise ValueError("Error model not implemented!")
        return PDE_error

    def _call_FD_neg_power(self, w, u, rho_j, z):
        """
        Higher order function that returns del_Castillo FD with given parameters
        """
        def FD_neg_power(rho):
            return z * ( (u*rho/rho_j)**(-w) + (1-rho/rho_j)**(-w) )**(-1/w)
        return FD_neg_power

    def _call_FD_exp(self, alpha, beta):
        def FD_exp(rho):
            """
            V(rho) = a*exp(-b*rho)
            """
            return alpha*np.exp(-beta*rho)*rho
        return FD_exp

    def transform_data(self, PDE_data, midas_data, fun_transform):
        """
        transformation to apply to both PDE_data and midas_data
        Parameters
        ---------
        PDE_data : list of PDE outputs
        midas_data : list of real data
        fun_transform : function (ex: np.sqrt)

        Returns
        -------
        PDE_Data : np.array of PDE outputs
        midas_data : np.array of real data
        """
        return np.array(list(map(fun_transform, np.array(PDE_data)))), np.array(list(map(fun_transform, midas_data)))


    def poisson_error(self, PDE_data, midas_data):
        """
        Returns the Poisson loss

        use math.log as np.log can't handle Long integers
        """
        poisson_scaling = list(map(math.log, list(map(math.factorial, midas_data.astype('int') )) ))
        poisson_array = np.array(PDE_data)*(-1) + midas_data*np.log(np.array(PDE_data)) - poisson_scaling
        return - np.sum(poisson_array) # need a minus here as there's another minus in MCMC log_likelihood.


    def squared_error(self, PDE_data, midas_data, loss_sd):
        """
        Returns the squared error loss
        """
        return np.sum(np.square(PDE_data - midas_data) * 0.5*(1/loss_sd**2)) # no minus sign as this is the loss. There's a minus sign in MCMC's log_likelihood function

    def get_residuals(self, BC_outlet=None, BC_inlet=None, **kwargs):
        """

        Parameters
        ----------
        BC_outlet, BC_inlet: Bool or ndarray

        Returns
        -------
        df_res : Dataframe with residuals and midas_data
            residuals: PDE_data-midas_data
            The rows are sorted by increasing values of midas_data
        """
        PDE_data, midas_data = self.solve_lwr(BC_outlet=BC_outlet, BC_inlet=BC_inlet, **kwargs)
        PDE_data, midas_data = self.transform_data(PDE_data=PDE_data, midas_data=midas_data, fun_transform=self.config.DATA_TRANSFORM)
        res_dict = {'residuals': PDE_data-midas_data, 'PDE_data': PDE_data}
        return pd.DataFrame(res_dict).sort_values(by='PDE_data')

    def get_data(self, variable):
        """
        Get density of flow data that is being used by LWR

        Parameters
        ----------
        variable: str
            Either 'flow' or 'density'
        """
        if variable == 'flow':
            data_array = np.genfromtxt(self.config.DATA_FLOW_PATH)
        elif variable == 'density':
            data_array = np.genfromtxt(self.config.DATA_DENSITY_PATH)
        return np.array([data_array[elem][2] for elem in range(np.shape(data_array)[0])])

    @property
    def df_FD_data(self):
        """
        Get dataframe of density and flow data used in LWR
        Keep only rows of flow and density that match (as there might be some
        missing data in flow or density)
        DataFrame columns: ['space', 'time', 'av flows', 'density_occ']
        """
        flow_data = np.genfromtxt(self.config.DATA_FLOW_PATH)
        den_data = np.genfromtxt(self.config.DATA_DENSITY_PATH)
        df_flow = pd.DataFrame(flow_data, columns=['space','time','flow'])
        df_density = pd.DataFrame(den_data, columns=['space','time','density'])
        df_both = pd.merge(df_flow, df_density, how='inner', on=['space','time'])
        # df_both.rename(columns={'density':'density_occ', 'flow': 'av flows'}, inplace=True)
        return df_both

    def high_res_BCs(self, BC_type):
        """
        Get high res BC for current dataset

        Parameters
        ----------
        BC_type: str
            'BC_outlet' or 'BC_inlet'
        """
        if self.config.data_array_dict['flow'] == 'data_array_70108_flow_49t.csv':
            if BC_type == "BC_outlet":
                return load_CSV_data(path='BCs/DS1_49t_raw_outlet_highres.csv')
            elif BC_type == "BC_inlet":
                return load_CSV_data(path='BCs/DS1_49t_raw_inlet_highres.csv')
            else:
                raise ValueError("'BC_type' should be either 'BC_outlet' or 'BC_inlet'")
        else:
            raise ValueError("Didn't create high resolution raw BCs for this dataset")
