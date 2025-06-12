import numpy as np
import sr_mcmc as srmc
import argparse
import config_lib as cl
import os
import deathTimesDataSet as dtds
import ast
import readResults as rr

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
    transform = config.getboolean('DEFAULT', 'transform')
    external_hazard = config.get('DEFAULT', 'external_hazard')
    data_dt = float(config.get('DEFAULT', 'data_dt', fallback=1))
    ndims = int(config.get('DEFAULT', 'ndims',fallback=4))

    if external_hazard == "None":
        external_hazard = np.inf
    else:
        external_hazard = float(external_hazard)


    h5_file_path = os.path.join(args.folder, f"{h5_file}_{args.index}.h5")
    ds = dtds.dsFromFile(data_file)
    ds.external_hazard = external_hazard
    seed_res = rr.readResultsFile(seed_file)
    res_df = seed_res[1]
    seed = rr.getTheta(res_df)
    if transform:
        seed = srmc.transform(seed)
    bins =srmc.get_bins_from_seed(seed, ndims =ndims, variations = variations)
    

    sampler = srmc.getSampler(nwalkers=nwalkers,
                                  num_mcmc_steps=num_mcmc_steps, dataSet= ds,back_end_file=h5_file_path,npeople=npeople,nsteps=nsteps,
                                  t_end=t_end,metric=metric,time_range=time_range, time_step_multiplier=time_step_multiplier,
                                  bins=bins, prior=prior,
                                  transformed=True, dt=data_dt
                                  )

    print(f"h5_file: {h5_file_path}")
    print(f"metric: {metric}")

if __name__ == "__main__":
    main()