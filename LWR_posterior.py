import numpy as np
from collections import OrderedDict
import os
from copy import deepcopy

import multiprocessing as mp
import time
from ou_process import OUProcess
from util import cut_prior_means, create_interpolate_prior_mean_fun, load_CSV_data, uniform_log_prob
from lwr.lwr_solver import LWR_Solver


# Define LWR
# ==========
config_dict = {'my_analysis_dir': '2018/June3_2018-DS1_del_Cast-rawBCs',
                'run_num': 1,
                'data_array_dict':
                        {'flow': 'data_array_70108_flow_49t.csv',
                        'density': 'data_array_70108_density_49t.csv'},
                'ratio_times_BCs': 40,
                      }
LWR = LWR_Solver(config_dict=config_dict)
lwr_final_time = deepcopy(LWR.final_time)
lwr_ratio_times_BCs = deepcopy(LWR.config.ratio_times_BCs)


# Define BC prior mean stuff
# ==========================

#  Load log-BC prior mean
data_array_dict = {'flow': 'data_array_70108_flow_49t.csv',
                    'density': 'data_array_70108_density_49t.csv'}

outlet_prior_mean_150 = load_CSV_data(path='prior_means/mean_outlet_150.csv')
inlet_prior_mean_150 = load_CSV_data(path='prior_means/mean_inlet_150.csv')
outlet_prior_mean = cut_prior_means(prior_mean=outlet_prior_mean_150, data_array_str=data_array_dict['flow'])
inlet_prior_mean = cut_prior_means(prior_mean=inlet_prior_mean_150, data_array_str=data_array_dict['flow'])
log_BC_prior_mean = {'BC_outlet': outlet_prior_mean,
                    'BC_inlet': inlet_prior_mean,
                    }

# Define OU prior
# ===============
OU_params = {'beta': 0.227812, 'dt': 1/lwr_ratio_times_BCs, 'sigma': 0.255809}
BC_len = lwr_final_time * lwr_ratio_times_BCs + 1
# BC_len = 200
ouprocess = OUProcess(N=BC_len, **OU_params)


# interpolate BCs when using a resolution higher than 1min
f_outlet = create_interpolate_prior_mean_fun(final_time=lwr_final_time, prior_mean_raw=log_BC_prior_mean['BC_outlet'])
f_inlet = create_interpolate_prior_mean_fun(final_time=lwr_final_time, prior_mean_raw=log_BC_prior_mean['BC_inlet'])
outlet_mean_highres = f_outlet(x=np.linspace(0, lwr_final_time, BC_len))
inlet_mean_highres = f_inlet(x=np.linspace(0, lwr_final_time, BC_len))
BC_prior_mean_OU_highres = OrderedDict()
BC_prior_mean_OU_highres['BC_outlet'] = outlet_mean_highres
BC_prior_mean_OU_highres['BC_inlet'] = inlet_mean_highres




def samplePrior():
    return ouprocess.sample()
precision_mat = deepcopy(ouprocess.Precision)

# KL transform:
_, evects = np.linalg.eigh(ouprocess.cov_prior)

def get_KL_weights(x):
    return evects.T @ x

def inverseKL(w):
    return evects @ w

def invG(u, a):
    phi = 1/(2*(np.sqrt(a) - 1/np.sqrt(a)))
    return (u/(2*phi) + 1/np.sqrt(a))**2

def sampleG(a):
    return invG(np.random.uniform(0,1), a)




prior_bounds = {'u': (1, 10), 'z': (100, 400), 'rho_j': (300, 800), 'w': (0.004, 10)}

def logPriorLWR(z, rho_j, u, w, logBC_outlet_centred, logBC_inlet_centred):
    lp_z = uniform_log_prob(theta=z, lower=prior_bounds['z'][0], upper=prior_bounds['z'][1])
    lp_rho_j = uniform_log_prob(theta=rho_j, lower=prior_bounds['rho_j'][0], upper=prior_bounds['rho_j'][1])
    lp_u = uniform_log_prob(theta=u, lower=prior_bounds['u'][0], upper=prior_bounds['u'][1])
    lp_w = uniform_log_prob(theta=w, lower=prior_bounds['w'][0], upper=prior_bounds['w'][1])

    lp_outlet = -0.5*np.linalg.multi_dot([logBC_outlet_centred, precision_mat, logBC_outlet_centred])
    lp_inlet = -0.5*np.linalg.multi_dot([logBC_inlet_centred, precision_mat, logBC_inlet_centred])

    return lp_z + lp_rho_j + lp_u + lp_w + lp_outlet + lp_inlet

def logLikelihoodLWR(z, rho_j, u, w, logBC_outlet_centred, logBC_inlet_centred):
    """
    Pass in FD and BCs

    Parameters
    ----------
    z, rho_j, u, w: floats
        FD parameters. Note that w is inverted
    logBC_outlet_centred, logBC_inlet_centred: ndarrays
        OU processes. These correspond to log(BCs) minus their mean

    """
    w_inv = 1/w
    BC_outlet = np.exp(logBC_outlet_centred + BC_prior_mean_OU_highres['BC_outlet'])
    BC_inlet = np.exp(logBC_inlet_centred + BC_prior_mean_OU_highres['BC_inlet'])
    # FD_array = np.array([z, rho_j, u, w])
    # le_mean = [160, 390, 6, 2]
    # return 10
    # return -0.5*np.linalg.multi_dot([FD_array-le_mean, FD_array-le_mean])
    return - LWR.loss_function(solver='lwr_del_Cast', z=z, rho_j=rho_j, u=u, w=w_inv, BC_outlet=BC_outlet, BC_inlet=BC_inlet)

def sampleFDPrior():
    "z, rho_j, u, w"
    FD_sample = np.array([np.random.uniform(*prior_bounds[k]) for k in ['z', 'rho_j','u','w'] ])
    return FD_sample

def AIES_step(args):
    """
    Do a AIES step for walker k using walker j to build the proposal.

    Parameters
    ----------
    args: list
        List with the following elements:
        Elements:
        ---------
        FD_k, outlet_k, inlet_k: ndarray
            Currrent state of walker k
        FD_j, outlet_j, inlet_j: ndarray
            Current state of walker j
        current_lp: float
            Current value of log-prior for walker k
        current_ll: float
            Current value of log-likelihood for walker k
        M_trunc: int
            Number of basis elements to keep for BC expanstion
        a_prop: float
            Step size for stretch move
        beta_temp: float
            Temperature for PT

    Returns
    -------
    args: list
        List with the following elements.
        Elements:
        ---------
        new_FD, new_outlet, new_inlet: ndarray
            New state of walker k
        newlp: float
            New value of log-prior
        newll: float
            New value of log-likelihood
        acceptBool: Bool
            Whether or not the proposal was accepted
    """
    FD_k, outlet_k, inlet_k, FD_j, outlet_j, inlet_j, current_lp, current_ll, M_trunc, a_prop, beta_temp = args
    # outlet weights
    w_Outletj0 = get_KL_weights(outlet_j)[-M_trunc:]
    w_Outlet_k = get_KL_weights(outlet_k)
    w_Outlet_k_start, w_Outlet_k_end = w_Outlet_k[:-M_trunc], w_Outlet_k[-M_trunc:]

    # inlet weights
    w_Inletj0 = get_KL_weights(inlet_j)[-M_trunc:]
    w_Inlet_k = get_KL_weights(inlet_k)
    w_Inlet_k_start, w_Inlet_k_end = w_Inlet_k[:-M_trunc], w_Inlet_k[-M_trunc:]

    Z = sampleG(a_prop)

    currentparm = np.concatenate([FD_k, w_Outlet_k_end, w_Inlet_k_end])
    otherparm = np.concatenate([FD_j, w_Outletj0, w_Inletj0])

    arrayProp = otherparm*(1-Z) + Z*currentparm
    FDProp, w_OutletProp, w_InletProp = arrayProp[:4], arrayProp[4: 4+M_trunc], arrayProp[-M_trunc:]

    outletProp = inverseKL(np.concatenate([w_Outlet_k_start, w_OutletProp]))
    inletProp = inverseKL(np.concatenate([w_Inlet_k_start, w_InletProp]))

    logPriorProp = logPriorLWR(*FDProp, outletProp, inletProp)
    logLikProp = logLikelihoodLWR(*FDProp, outletProp, inletProp)

    log_alpha = (2*M_trunc+4-1)*np.log(Z) + logPriorProp + beta_temp*logLikProp - current_lp - beta_temp*current_ll

    if log_alpha > (-np.random.exponential()):
        return [FDProp, outletProp, inletProp, logPriorProp, logLikProp, True]
    else:
        return [FD_k, outlet_k, inlet_k, current_lp, current_ll, False]


def pCNStep(args):
    """
    pCN step

    Parameters
    ----------
    args: list
        List with the following elements:
        Elements:
        ---------
        FD_k, outlet_k, inlet_k: ndarray
                        Currrent state of walker k
        BC_type: str
            either 'BC_outlet' or 'BC_inlet'
        omega: float
            Step size for pCN
        current_lp: float
            Current value of log-prior for walker k
        current_ll: float
            Current value of log-likelihood for walker k
        beta_temp: float
            Temperature for PT
    Returns
    -------
    args: list
        List with the following elements.
        Elements:
        ---------
        new_outlet, new_inlet: ndarray
            New value of outlet and inlet
        newlp: float
            New value of log-prior
        newll: float
            New value of log-likelihood
        acceptBool: Bool
            Whether or not the proposal was accepted
    """
    FD_k, outlet_k, inlet_k, BC_type, omega, current_lp, current_ll, beta_temp = args
    if BC_type == 'BC_outlet':
        outletProp = np.sqrt(1-omega**2)*outlet_k + omega*samplePrior()
        inletProp = inlet_k
    elif BC_type == 'BC_inlet':
        outletProp = outlet_k
        inletProp = np.sqrt(1-omega**2)*inlet_k + omega*samplePrior()
    else:
        raise ValueError("BC_type is either 'BC_outlet' or 'BC_inlet'")

    logLikProp = logLikelihoodLWR(*FD_k, outletProp, inletProp)
    log_alpha = beta_temp*(logLikProp - current_ll)
    if log_alpha > (-np.random.exponential()):
        newlp = logPriorLWR(*FD_k, outletProp, inletProp)
        return [outletProp, inletProp,  newlp, logLikProp, True]
    else:
        return [outlet_k, inlet_k, current_lp, current_ll, False]


def temperature_swap(currentFD, currentOutlet, currentInlet, currentLogLik, currentLogPrior, betas):
    """
    Temperature swap

    Parameters:
    ----------
    currentFD, currentOutlet, currentInlet: ndarray
        Current parameters for each walker and temperature
    currentLogLik, currentLogPrior: ndarray
        Current log-likelihood and log-prior for each walker and temperature
    betas: ndarray
        List of inverse temperatures


    Returns:
    --------
    currentFD, currentOutlet, currentInlet: ndarray
        Current parameters for each walker and temperature
    currentLogLik, currentLogPrior: ndarray
        Current log-likelihood and log-prior for each walker and temperature
    swaps_accepted: ndarray
        Number of accepted swaps for each temperature
        swaps_accepted[0] is the number of accepted swaps for the untempered chains
    """
    num_temps, Lwalkers, _ = np.shape(currentFD)
    swaps_accepted = np.zeros(num_temps-1)

    for i in range(num_temps-1, 0, -1):
        bi = betas[i]
        bi1 = betas[i-1]

        iperm = np.random.permutation(Lwalkers)
        i1perm = np.random.permutation(Lwalkers)

        raccept = np.log(np.random.uniform(size=Lwalkers))
        paccept = (bi1 - bi) * (currentLogLik[i, iperm] - currentLogLik[i-1, i1perm])

        accepts = paccept>raccept
        swaps_accepted[i-1] = np.sum(accepts)

        FD_temp = np.copy(currentFD[i, iperm[accepts], :])
        outlet_temp = np.copy(currentOutlet[i, iperm[accepts], :])
        inlet_temp = np.copy(currentInlet[i, iperm[accepts], :])
        logLik_temp = np.copy(currentLogLik[i, iperm[accepts]])
        logPrior_temp = np.copy(currentLogPrior[i, iperm[accepts]])

        currentFD[i, iperm[accepts], :] = currentFD[i-1, i1perm[accepts], :]
        currentOutlet[i, iperm[accepts], :] = currentOutlet[i-1, i1perm[accepts], :]
        currentInlet[i, iperm[accepts], :] = currentInlet[i-1, i1perm[accepts], :]
        currentLogLik[i, iperm[accepts]] = currentLogLik[i-1, i1perm[accepts]]
        currentLogPrior[i, iperm[accepts]] = currentLogPrior[i-1, i1perm[accepts]]

        currentFD[i-1, i1perm[accepts], :] = FD_temp
        currentOutlet[i-1, i1perm[accepts], :] = outlet_temp
        currentInlet[i-1, i1perm[accepts], :] = inlet_temp
        currentLogLik[i-1, i1perm[accepts]] = logLik_temp
        currentLogPrior[i-1, i1perm[accepts]] = logPrior_temp

    return currentFD, currentOutlet, currentInlet, currentLogLik, currentLogPrior, swaps_accepted


if BC_len != 200:
    IC_outlet = np.log(np.genfromtxt("data/MCMC_ICs/outlet_2K.txt")) - BC_prior_mean_OU_highres['BC_outlet']
    IC_inlet = np.log(np.genfromtxt("data/MCMC_ICs/inlet_2K.txt")) - BC_prior_mean_OU_highres['BC_inlet']
    IC_FD = {'z': 175.64524045695617,
     'rho_j': 379.3928422197564,
     'u': 3.99525422856635,
     'w': 0.19811128675617157,
      "logBC_outlet_centred": IC_outlet,
    "logBC_inlet_centred": IC_inlet
         }


def generate_LWR_ICs():
    """
    Generate initial conditions for a walker in AIES: start from a posterior sample and add noise

    Returns 3 numpy arrays for FDs, outlet, and inlet.
    """
    FD_sds = [10, 20, 1]
    FD_noise = {k: np.random.normal(v, lesd) for lesd, (k,v) in zip(FD_sds, IC_FD.items()) if k in ['z','rho_j','u']}
    FD_noise['w'] = np.exp(np.random.normal(0, 0.5))*IC_FD['w']

    while any([not prior_bounds[elem][0]<FD_noise[elem]<prior_bounds[elem][1] for elem in ['z','rho_j','u','w']]):
        FD_noise = {k: np.random.normal(v, lesd) for lesd, (k,v) in zip(FD_sds, IC_FD.items()) if k in ['z','rho_j','u']}
        FD_noise['w'] = np.exp(np.random.normal(0, 0.5))*IC_FD['w']

    newFDs = np.array([v for k,v in FD_noise.items() if k in ['z','rho_j','u','w']])
    omega = 0.2
    newOutlet = np.sqrt(1-omega**2)*IC_FD['logBC_outlet_centred'] + omega*samplePrior()
    newInlet = np.sqrt(1-omega**2)*IC_FD['logBC_inlet_centred'] + omega*samplePrior()
    return newFDs, newOutlet, newInlet

if BC_len == 200:
    def generate_LWR_ICs():
        newFDs = sampleFDPrior()
        newOutlet = samplePrior()
        newInlet = samplePrior()
        return newFDs, newOutlet, newInlet

def acceptance_rates(num_acceptsPCN_outlet, num_PCN_outlet, num_acceptsPCN_inlet, num_PCN_inlet, num_acceptsAIES, num_AIES, num_acceptsSwaps=None, num_Swaps=None):
    try:
        acceptance_ratepCN_Outlet = num_acceptsPCN_outlet / (num_PCN_outlet) * 100
    except ZeroDivisionError:
        acceptance_ratepCN_Outlet = 0
    try:
        acceptance_ratepCN_Inlet = num_acceptsPCN_inlet / (num_PCN_inlet) * 100
    except ZeroDivisionError:
        acceptance_ratepCN_Inlet = 0
    try:
        acceptance_rateAIES = num_acceptsAIES / (num_AIES) * 100
    except ZeroDivisionError:
        acceptance_rateAIES = 0
    if num_acceptsSwaps is None:
        return acceptance_ratepCN_Outlet, acceptance_ratepCN_Inlet, acceptance_rateAIES
    else:
        try:
            acceptance_rateSwaps = num_acceptsSwaps / (num_Swaps) * 100
        except ZeroDivisionError:
            acceptance_rateSwaps = 0
        return acceptance_ratepCN_Outlet, acceptance_ratepCN_Inlet, acceptance_rateAIES, acceptance_rateSwaps
