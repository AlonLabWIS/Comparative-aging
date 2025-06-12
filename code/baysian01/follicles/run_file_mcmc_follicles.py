import numpy as np
#adding baysian01 folder directory to path
import sys
sys.path.append("baysian01")
import sr_mcmc as srmc
import argparse
import config_lib as cl
import os
import deathTimesDataSet as dtds
import ast
import readResults as rr
import readFollicles as rf
import SRclass_follicles as srf

def main():
    parser = argparse.ArgumentParser(description="A script that processes command line arguments.")
    parser.add_argument("config_path", type=str, help="the path to the config file to get params from.")
    parser.add_argument("folder", type=str, help="The path of the output h5 folder")
    parser.add_argument("index", type=int, help="The index of array job")

    
    args = parser.parse_args()
    config = cl.read_configs(args.config_path)
    nsteps = int(config.get('DEFAULT', 'nsteps'))
    npeople = int(config.get('DEFAULT', 'npeople'))
    t_end = int(config.get('DEFAULT', 't_end'))
    nwalkers = int(config.get('DEFAULT', 'nwalkers'))
    nsteps = int(config.get('DEFAULT', 'nsteps'))
    h5_file = config.get('DEFAULT', 'h5_file_name')
    num_mcmc_steps = int(config.get('DEFAULT', 'n_mcmc_steps'))
    metric = config.get('DEFAULT', 'metric')
    time_range = ast.literal_eval(config.get('DEFAULT', 'time_range'))
    time_step_multiplier = int(config.get('DEFAULT', 'time_step_multiplier'))
    data_file = config.get('DEFAULT', 'data_file')
    seed_file = config.get('DEFAULT', 'seed_file')
    variations = ast.literal_eval(config.get('DEFAULT', 'variations'))
    prior = int(config.get('DEFAULT', 'prior'))
    external_hazard = config.get('DEFAULT', 'external_hazard')
    initial_follicles_mean = config.get('DEFAULT', 'initial_follicles_mean', fallback=12.692312588793072)
    initial_follicles_std = config.get('DEFAULT', 'initial_follicles_std', fallback=0.441272921587803)
    log_initial_follicles = config.get('DEFAULT', 'log_initial_follicles', fallback='True')
    params = config.get('DEFAULT', 'params',fallback='[0.49275,54.75,51.83,17]')
    data_dt = config.getint('DEFAULT', 'data_dt', fallback=2)
    ndims = int(config.get('DEFAULT', 'ndims',fallback=3))

    if initial_follicles_mean is not None:
        initial_follicles_mean = float(initial_follicles_mean)
    if initial_follicles_std is not None:
        initial_follicles_std = float(initial_follicles_std)
    if log_initial_follicles is not None:
        log_initial_follicles = ast.literal_eval(log_initial_follicles)
    if params is not None:
        params = ast.literal_eval(params)

    if external_hazard == "None":
        external_hazard = np.inf


    h5_file_path = os.path.join(args.folder, f"{h5_file}_{args.index}.h5")
    ds = rf.folliclesFromFile(data_file)
    ds.external_hazard = external_hazard
    seed = rf.readSeedFollicles(seed_file)
    bins  =srmc.get_bins_from_seed(seed, variations = variations, ndims=ndims)
    set_params = {'eta': params[0], 'beta': params[1], 'epsilon': params[2], 'xc': params[3],'external_hazard': external_hazard}


    sampler = srmc.getSampler(nwalkers=nwalkers,
                                  num_mcmc_steps=num_mcmc_steps, dataSet= ds,back_end_file=h5_file_path,npeople=npeople,nsteps=nsteps,
                                  t_end=t_end,metric=metric,time_range=time_range, time_step_multiplier=time_step_multiplier,
                                  bins=bins, prior=prior, params=params, dt=data_dt,set_params=set_params, ndim=ndims, model_func= srf.model,
                                  initial_follicles_mean=initial_follicles_mean, initial_follicles_std=initial_follicles_std,
                                  log_initial_follicles=log_initial_follicles
                                  )

    print(f"h5_file: {h5_file_path}")
    print(f"metric: {metric}")
    print(f"data_times_discretisation: {data_dt}")

if __name__ == "__main__":
    main()