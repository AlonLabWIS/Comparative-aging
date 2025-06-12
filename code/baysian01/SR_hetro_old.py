"""
This class inherits from the SR_lf class but adds an option for heterogeneity in the model parameters - eta, beta, epsilon, and xc. The heterogeneity is modeled as a normal distribution with a given mean and standard deviation.  
"""

import numpy as np
import SRmodellib_lifelines as srl
from joblib import Parallel, delayed
from numba import jit
import os


jit_nopython = True


class SR_hetro(srl.SR_lf):
    def __init__(self, eta, beta, kappa, epsilon, xc,
                  npeople, nsteps, t_end, eta_var =0, beta_var =0, epsilon_var =0, xc_var =0, t_start = 0,
                    tscale = 'years',external_hazard=np.inf,time_step_multiplier=1, parallel =False, bandwidth=3):
        """
        This function initializes the SR_hetro class. 
        """
        self.eta_var = eta_var
        self.beta_var = beta_var
        self.epsilon_var = epsilon_var
        self.xc_var = xc_var
        super().__init__(eta, beta, kappa, epsilon, xc,
                  npeople, nsteps, t_end, t_start = 0,
                    tscale = 'years',external_hazard=external_hazard, time_step_multiplier = time_step_multiplier, parallel=parallel, bandwidth=bandwidth)
        

    
    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1]-self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        if self.parallel:
            death_times=death_times_accelerator2(s,dt,t,self.eta,self.eta_var,self.beta,self.beta_var,self.kappa,self.epsilon, self.epsilon_var,self.xc,self.xc_var,sdt,self.npeople,self.external_hazard,self.time_step_multiplier)
        else:
            death_times=death_times_accelerator2(s,dt,t,self.eta,self.eta_var,self.beta,self.beta_var,self.kappa,self.epsilon, self.epsilon_var,self.xc,self.xc_var,sdt,self.npeople,self.external_hazard,self.time_step_multiplier)

        return np.array(death_times)
    

def death_times_accelerator2(s,dt,t,eta,eta_var,beta,beta_var,kappa,epsilon,epsilon_var,xc,xc_var,sdt,npeople,external_hazard = np.inf,time_step_multiplier = 1):
    @jit(nopython=jit_nopython)
    def calculate_death_times(npeople, s, dt, t, eta0,eta_var,beta0,beta_var,kappa,epsilon0,epsilon_var,xc0,xc_var, sdt, external_hazard,time_step_multiplier):
        death_times = []
        for i in range(npeople):
            died = False
            x = 0
            j = 0
            ndt = dt/time_step_multiplier
            nsdt = np.sqrt(ndt)
            chance_to_die_externally = np.exp(-external_hazard)*ndt
            eta = np.random.normal(loc = eta0,scale = eta_var)
            beta = np.random.normal(loc = beta0,scale = beta_var)
            epsilon = np.random.normal(loc = epsilon0,scale = epsilon_var)
            xc = np.random.normal(loc = xc0,scale = xc_var)
            while j in range(s - 1) and x < xc and not died:
                for i in range(time_step_multiplier):
                    noise = np.sqrt(2*epsilon)*np.random.normal(loc = 0,scale = 1)
                    x = x+ndt*(eta*(t[j]+i*ndt)-beta*x/(x+kappa))+noise*nsdt
                    x = np.maximum(x, 0)
                    if np.random.uniform(0,1)<chance_to_die_externally:
                        x = xc
                    if x>=xc:
                        died = True
                j += 1
            if died:
                death_times.append(j * dt)
        return death_times

    n_jobs = os.cpu_count()
    npeople_per_job = npeople // n_jobs
    results = Parallel(n_jobs=n_jobs)(delayed(calculate_death_times)(
        npeople_per_job, s, dt, t, eta,eta_var, beta,beta_var, kappa, epsilon,epsilon_var, xc,xc_var, sdt, external_hazard,time_step_multiplier
    ) for _ in range(n_jobs))

    death_times = [dt for sublist in results for dt in sublist]
    return death_times

@jit(nopython=jit_nopython)
def death_times_accelerator(s,dt,t,eta0,eta_var,beta0,beta_var,kappa,epsilon0,epsilon_var,xc0,xc_var,sdt,npeople,external_hazard = np.inf, time_step_multiplier = 1):
    death_times = []
    for i in range(npeople):
        x=0
        j=0
        ndt = dt/time_step_multiplier
        nsdt = sdt/np.sqrt(time_step_multiplier)
        chance_to_die_externally = np.exp(-external_hazard)*ndt
        eta = np.random.normal(loc = eta0,scale = eta_var)
        beta = np.random.normal(loc = beta0,scale = beta_var)
        epsilon = np.random.normal(loc = epsilon0,scale = epsilon_var)
        xc = np.random.normal(loc = xc0,scale = xc_var)
        while j in range(s-1) and x<xc:
            for i in range(time_step_multiplier):
                noise = np.sqrt(2*epsilon)*np.random.normal(loc = 0,scale = 1)
                x = x+ndt*(eta*t[j]*(1+i*ndt)-beta*x/(x+kappa))+noise*nsdt
                x = np.maximum(x, 0)
                if np.random.uniform(0,1)<chance_to_die_externally:
                    x = xc
                if x>=xc:
                    break
            j+=1
        if x>=xc:
            death_times.append(j*dt)

    return death_times