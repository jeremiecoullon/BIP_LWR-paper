import numpy as np
import os
import multiprocessing as mp
import time

from pathlib import Path
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from LWR_posterior import samplePrior, sampleFDPrior, logPriorLWR, logLikelihoodLWR
from LWR_posterior import AIES_step, pCNStep, BC_len, generate_LWR_ICs, acceptance_rates, temperature_swap
from util import gen_run_setting_str, save_chain_to_file, save_current_samples

"""
To run: `python -W ignore`
"""

N_MCMC = 34000

Comments = "LWR PT: run 34K (part3 of run19). Save currentSamples as well as untempered samples"
Lwalkers = 13
M_trunc = 4
a_prop = 2
omega_outlet_list, omega_inlet_list = [0.078, 0.09, 0.11, 0.15], [0.155, 0.17, 0.2, 0.25]
thin_samples = 100
move_probs = [0.25, 0.125, 0.125, 0.5] # AIES, pCN outlet, pCN inlet, swap

betas = [1, 0.76, 0.58, 0.44]


dir_name = 'Run-19May2020-PT-Part3'
# save_chains = True
# save_to_S3 = True

save_chains = False
save_to_S3 = False

# save_checkpoints = [10, 10000, 25000, 40000, 50000, 60000, 80000, 90000]
save_checkpoints = np.arange(0, N_MCMC, 1000)
# ================
# ================
# put all run parametres in a dictionary
config_dict = {"N_MCMC": N_MCMC, "Comments": Comments, "Lwalkers": Lwalkers, "M_trunc":M_trunc,
    "a_prop": a_prop, "omega_outlet_list": omega_outlet_list, "omega_inlet_list": omega_inlet_list,
    "thin_samples": thin_samples, "move_probs": move_probs, "betas": betas, "dir_name": dir_name,
    "save_chains": save_chains, "save_to_S3": save_to_S3}
# create path for MCMC outputs
parent_path = Path(__file__).parent.absolute()
Path(os.path.join(parent_path, f'outputs/{dir_name}')).mkdir(exist_ok=True)
Path(os.path.join(parent_path, f'outputs/{dir_name}/currentSamples')).mkdir(exist_ok=True)

assert N_MCMC % thin_samples == 0
N_saved = int(N_MCMC/thin_samples)
num_temps = len(betas)

# initialise MCMC: N MCMC iterations, num_temps temperatures, L walkers, BC_len: resolution of function

currentOutlet = np.zeros((num_temps, Lwalkers, BC_len))
currentInlet = np.zeros((num_temps, Lwalkers, BC_len))
currentFD = np.zeros((num_temps, Lwalkers, 4))

# ==============
# Initialise MCMC
# ==============
print("Initialising walkers for all temperatures..")
for i in range(num_temps):
    L_ICs = np.array([generate_LWR_ICs() for e in range(Lwalkers)])
    currentOutlet[i, :, :] = np.array([L_ICs[e][1] for e in range(Lwalkers)])
    currentInlet[i, :, :] = np.array([L_ICs[e][2] for e in range(Lwalkers)])
    currentFD[i, :, :] = np.array([L_ICs[e][0] for e in range(Lwalkers)])


# ==============
# To run MCMC from the end of another chain
# ==============
# print("Loading samples from the end of the last chain..")
# currentS_dir = "MCMC_IC/Run19-part2-currentSamples/"
# FD_current = {}
# Outlet_current = {}
# Inlet_current = {}
#
# for t in range(num_temps):
#     FD_current[t] = np.genfromtxt(currentS_dir+f"FD_temp{t}.txt")
#     Outlet_current[t] = np.genfromtxt(currentS_dir+f"Outlet_temp{t}.txt")
#     Inlet_current[t] = np.genfromtxt(currentS_dir+f"Inlet_temp{t}.txt")
# for i in range(num_temps):
#     currentOutlet[i, :, :] = Outlet_current[i]
#     currentInlet[i, :, :] = Inlet_current[i]
#     currentFD[i, :, :] = FD_current[i]
# print("Done.")
# ==============
# ==============


currentLogPrior = np.zeros((num_temps, Lwalkers))
currentLogLik = np.zeros((num_temps, Lwalkers))

# keep samples for untempered chains
samplesOutlet = np.zeros((N_saved, Lwalkers, BC_len))
samplesInlet = np.zeros((N_saved, Lwalkers, BC_len))
samplesFD = np.zeros((N_saved, Lwalkers, 4))

logPriorList = np.zeros((N_saved, Lwalkers))
logLikList = np.zeros((N_saved, Lwalkers))


for t in range(num_temps):
    for k in range(Lwalkers):
        currentLogPrior[t, k] = logPriorLWR(*currentFD[t, k, :], currentOutlet[t, k,:], currentInlet[t, k,: ])
        currentLogLik[t, k] = logLikelihoodLWR(*currentFD[t, k, :], currentOutlet[t, k,:], currentInlet[t, k,: ])
        if t==0:
            logPriorList[0, k] = currentLogPrior[0, k]
            logLikList[0, k] = currentLogLik[0, k]
            samplesOutlet[0, k, :] = currentOutlet[0, k, :]
            samplesInlet[0, k, :] = currentInlet[0, k, :]
            samplesFD[0, k, :] = currentFD[0, k, :]
        else:
            pass

# only track acceptance rates for the untempered chains
num_acceptsPCN_outlet = 0
num_acceptsPCN_inlet = 0
num_acceptsAIES = 0
# track accepts for (num_temps-1) pairs of swaps
num_acceptsSwaps = np.zeros(num_temps-1)

num_PCN_outlet, num_PCN_inlet, num_AIES, num_Swaps = 0,0,0,0


if __name__ == "__main__":


    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)

    start = time.time()

    start_subtime = time.time()
    print(f"""\nRunning function space AIES for {N_MCMC} iterations (keeping every {thin_samples} samples) and {Lwalkers} walkers.
    M_trunc={M_trunc}, and proposal variance a={a_prop}. Using {num_cores} cores.
    Saving samples to text files at iterations {save_checkpoints}.\n""")
    for i in range(1, N_MCMC):

        which_move = np.random.choice([1,2,3,4], p=move_probs)

        if which_move == 1:
            # AIES
            num_AIES += Lwalkers
            mylist = list(range(Lwalkers))
            np.random.shuffle(mylist)
            halfL = int(Lwalkers / 2)
            S1, S2 = mylist[:halfL], mylist[halfL:Lwalkers]

            Slist = [S1, S2]
            for idxS in [0,1]:
                S_current = Slist[idxS]
                S_other = Slist[idxS-1]

                S_arg_list = []
                for kk in S_current:
                    for tt in range(num_temps):
                        j0 = np.random.choice(S_other)
                        arg_list = [currentFD[tt, kk, :], currentOutlet[tt, kk,:], currentInlet[tt, kk,:],
                                    currentFD[tt, j0, :], currentOutlet[tt, j0,:], currentInlet[tt, j0,:],
                                    currentLogPrior[tt, kk], currentLogLik[tt, kk], M_trunc, a_prop,
                                    betas[tt]]
                        S_arg_list.append(arg_list)

                results = pool.map(AIES_step, S_arg_list)

                S_t_list = [(a,b) for a in S_current for b in range(num_temps)]
                for (S1_idx, tt), (a,b,c,d,e,f) in zip(S_t_list, results):
                    currentFD[tt, S1_idx, :] = a
                    currentOutlet[tt, S1_idx,:] = b
                    currentInlet[tt, S1_idx,:] = c
                    currentLogPrior[tt, S1_idx] = d
                    currentLogLik[tt, S1_idx] = e
                    if tt==0:
                        num_acceptsAIES += int(f)


        elif which_move == 2:
            # pcN for outlet
            num_PCN_outlet += Lwalkers
            pCN_arg_list = []
            for k in range(Lwalkers):
                for tt in range(num_temps):
                    pCN_arg_list.append([currentFD[tt, k,:], currentOutlet[tt, k,: ], currentInlet[tt, k,:], "BC_outlet",
                            omega_outlet_list[tt], currentLogPrior[tt, k], currentLogLik[tt, k], betas[tt]])

            results = pool.map(pCNStep, pCN_arg_list)

            t_L_list = [(a,b) for a in range(Lwalkers) for b in range(num_temps)]
            for (k, tt), (outlet_new, inlet_new, new_lp, new_ll, acceptBool) in zip(t_L_list, results):
                currentOutlet[tt, k,:] = outlet_new
                currentInlet[tt, k,:] = inlet_new
                currentLogPrior[tt, k] = new_lp
                currentLogLik[tt, k] = new_ll
                if tt==0:
                    num_acceptsPCN_outlet += int(acceptBool)

        elif which_move == 3:
            # pCN for inlet
            num_PCN_inlet += Lwalkers
            pCN_arg_list = []
            for k in range(Lwalkers):
                for tt in range(num_temps):
                    pCN_arg_list.append([currentFD[tt, k,:], currentOutlet[tt, k,: ], currentInlet[tt, k,:], "BC_inlet",
                            omega_inlet_list[tt], currentLogPrior[tt, k], currentLogLik[tt, k], betas[tt]])

            results = pool.map(pCNStep, pCN_arg_list)

            t_L_list = [(a,b) for a in range(Lwalkers) for b in range(num_temps)]
            for (k, tt), (outlet_new, inlet_new, new_lp, new_ll, acceptBool) in zip(t_L_list, results):
                currentOutlet[tt, k,:] = outlet_new
                currentInlet[tt, k,:] = inlet_new
                currentLogPrior[tt, k] = new_lp
                currentLogLik[tt, k] = new_ll
                if tt==0:
                    num_acceptsPCN_inlet += int(acceptBool)

        elif which_move == 4:
            num_Swaps += Lwalkers
            a,b,c,d,e,f = temperature_swap(currentFD, currentOutlet, currentInlet, currentLogLik, currentLogPrior, betas)
            currentFD[:,:,:] = a
            currentOutlet[:,:,:] = b
            currentInlet[:,:,:] = c
            currentLogLik[:,:] = d
            currentLogPrior[:,:] = e
            num_acceptsSwaps += f

        if i%thin_samples == 0:
            i_save = int(i/thin_samples)
            samplesOutlet[i_save, :, :] = currentOutlet[0, :, :]
            samplesInlet[i_save, :, :] = currentInlet[0, :, :]
            samplesFD[i_save, :, :] = currentFD[0, :, :]
            logLikList[i_save, :] = currentLogLik[0, :]
            logPriorList[i_save, :] = currentLogPrior[0, :]

        # if i%500 == 0:
        if i%1 == 0:
            end_subtime = time.time()
            print(f"Iteration {i}/{N_MCMC}")
            print(f"The last 500 iterations took: {((end_subtime-start_subtime)/60):.2f} min")
            start_subtime = time.time()

        # if i in save_checkpoints:
        if i % 1000 == 0:
            acceptance_ratepCN_Outlet, acceptance_ratepCN_Inlet, acceptance_rateAIES, acceptance_rateSwaps = acceptance_rates(num_acceptsPCN_outlet, num_PCN_outlet,
                                                                                num_acceptsPCN_inlet, num_PCN_inlet, num_acceptsAIES, num_AIES,
                                                                                num_acceptsSwaps, num_Swaps)
            config_dict["acceptance_ratepCN_Outlet"] = acceptance_ratepCN_Outlet
            config_dict["acceptance_ratepCN_Inlet"] = acceptance_ratepCN_Inlet
            config_dict["acceptance_rateAIES"] = acceptance_rateAIES
            config_dict["acceptance_rateSwaps"] = acceptance_rateSwaps
            if save_chains:
                print("Saving chains..")
                for w_num in range(Lwalkers):
                    save_chain_to_file(config_dict, samplesOutlet[:,w_num,:], samplesInlet[:,w_num,:], samplesFD[:,w_num,:], logLikList+logPriorList, w_num)
                save_current_samples(config_dict, i, currentOutlet, currentInlet, currentFD)
                print("Done saving.")


    end = time.time()

    acceptance_ratepCN_Outlet, acceptance_ratepCN_Inlet, acceptance_rateAIES, acceptance_rateSwaps = acceptance_rates(num_acceptsPCN_outlet, num_PCN_outlet,
                                                                        num_acceptsPCN_inlet, num_PCN_inlet, num_acceptsAIES, num_AIES,
                                                                        num_acceptsSwaps, num_Swaps)
    config_dict["acceptance_ratepCN_Outlet"] = acceptance_ratepCN_Outlet
    config_dict["acceptance_ratepCN_Inlet"] = acceptance_ratepCN_Inlet
    config_dict["acceptance_rateAIES"] = acceptance_rateAIES
    config_dict["acceptance_rateSwaps"] = acceptance_rateSwaps
    print(f"Acceptance rate for pCN Outlet: {acceptance_ratepCN_Outlet:.1f}%")
    print(f"Acceptance rate for pCN Inlet: {acceptance_ratepCN_Inlet:.1f}%")
    print(f"Acceptance rate for AIES: {acceptance_rateAIES:.1f}%")
    print(f"Acceptance rate for swaps: {[round(e, 1) for e in acceptance_rateSwaps]}%")


    if save_chains:
        print("Saving chains..")
        for w_num in range(Lwalkers):
            save_chain_to_file(config_dict, samplesOutlet[:,w_num,:], samplesInlet[:,w_num,:], samplesFD[:,w_num,:], logLikList+logPriorList, w_num)
        save_current_samples(config_dict, i, currentOutlet, currentInlet, currentFD)
        print("Done saving.")

    print("\nDone sampling.")
    print(f"\nRuning time {end-start:.2f}s")
