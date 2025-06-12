import SRmodellib_lifelines as srl
import numpy as np
from numba import jit
from joblib import Parallel, delayed
import deathTimesDataSet as dtds
import matplotlib.pyplot as plt
import os
import sr_mcmc as srmc

jit_nopython = True

"""
After implementing your class, change sr_mcmc.model so it calls your class instead of the default one and uses your metric function.
"""


class SR_Follicles(srl.SR_lf):
    def __init__(self, eta, beta, kappa, epsilon, xc, lambda0, lambda1, lambda2, npeople, nsteps, t_end, t_start=0, tscale='years', external_hazard=np.inf, time_step_multiplier=1, parallel=False, bandwidth=3, heun=False,
                 initial_follicles_mean = 12.692312588793072, initial_follicles_std = 0.441272921587803, log_initial_follicles = True):
        """
        This class is a subclass of SR_lf from SRmodellib_lifelines.py. It is used to calculate the distribution of follicle counts
        """
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.initial_follicles_mean = initial_follicles_mean
        self.initial_follicles_std = initial_follicles_std
        self.log_initial_follicles = log_initial_follicles
        
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, t_start, tscale, external_hazard, time_step_multiplier, parallel, bandwidth, heun)




    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1]-self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
  
        if self.parallel:
            death_times, events,follicles =death_times_accelerator2(s,dt,t,self.eta,self.beta,self.kappa,self.epsilon,self.xc,self.lambda0, self.lambda1, self.lambda2,self.t_end,sdt,self.npeople,self.external_hazard,self.time_step_multiplier, self.initial_follicles_mean, self.initial_follicles_std, self.log_initial_follicles)
        else:
            death_times, events, follicles =death_times_accelerator(s,dt,t,self.eta,self.beta,self.kappa,self.epsilon,self.xc,self.lambda0, self.lambda1, self.lambda2,self.t_end,sdt,self.npeople,self.external_hazard,self.time_step_multiplier, self.initial_follicles_mean, self.initial_follicles_std, self.log_initial_follicles)
        follicles =np.array(follicles)
        self.follicles = follicles
        follicles[follicles==-np.inf] = np.nan 
        follicles[follicles<0] = 0
        return np.array(death_times), np.array(events)
    

    def getFolliclesStats(self,log =False):
        #returns the mean and std for all times for the number of follicles.
        if log:
            follicles = np.log(self.follicles)
        else:
            follicles = self.follicles
        means = np.nanmean(follicles, axis=0)
        stds = np.nanstd(follicles, axis=0)
    
        return means, stds
    
    def plotFollicles(self, ax=None,nstds=1, ntrajectories=10, randomize_index=False,traj_color='grey',plot_color='blue',logstats=False, **kwargs): 
        if self.follicles is None:
            raise ValueError('No follicles were saved or calculated. Run the simulation first.')
        if ax is None:
            fig, ax = plt.subplots()
        if randomize_index:
            indices = np.random.choice(self.npeople, ntrajectories, replace=False)
        else:
            indices = np.arange(ntrajectories)
        for i in indices:
            ax.plot(self.follicles[i, :],color=traj_color, alpha=0.1)
        means, stds = self.getFolliclesStats(log=logstats)
        if logstats:
            errors = [np.exp(means-(stds*nstds)),np.exp(means+(stds*nstds))]
            means = np.exp(means)
        else:
            errors = [means-(stds*nstds),means+(stds*nstds)]
        ax.plot(means, color=plot_color, **kwargs)
        ax.fill_between(np.arange(len(means)), errors[0], errors[1], color=plot_color, alpha=0.3)
        return ax
    
    def getFoliiclesDists(self, log_scale=True, interval =2):
        #returns a kde of the follicles at each time point (with distanced interval between time points)
        #the kde is calculated using the follicles of all individuals at each time point and the KDEMultivariate from statsmodels function 
        #the function returns a dictionary with the kde for each time point
        from statsmodels.nonparametric.kernel_density import KDEMultivariate
        from scipy.stats import gaussian_kde
        follicles = self.follicles.copy()
        t = np.arange(0,self.t_end)
        follicles_dists = {}
        for i in range(0,len(t),interval):
            f=follicles[:,i]
            f = f[~np.isnan(f)]
            if log_scale:
                f = np.log(f)
            # f = np.log(f)

            follicles_dists[i]=(KDEMultivariate(data=f, var_type='c'))
            # follicles_dists[i]=gaussian_kde(f) 
        return follicles_dists
    
    def sim_to_follicle_ds(self, n, interval =2):
        #chooses n random individuals from the simulation and chooses a random time point for each individual
        #the time point should be a multiple of interval.
        #returns the follicle count at that time point for each individual as a 2D ndarray of shape (n,2) 
        #the first column is the time point and the second column is the follicle count
        if self.follicles is None:
            raise ValueError('No follicles were saved or calculated. Run the simulation first.')
        indices = np.random.choice(self.npeople, n, replace=False)
        time_points = np.random.choice(np.arange(0, self.t_end, interval), n, replace=True)
        follicles_ds = np.zeros((n,2))
        for i in range(n):
            follicles_ds[i,0] = time_points[i]
            follicles_ds[i,1] = self.follicles[indices[i], time_points[i]]
            #make sure that the follicle count is not nan
            while np.isnan(follicles_ds[i,1]):
                time_points[i] = np.random.choice(np.arange(0, self.t_end, interval), 1, replace=True)
                follicles_ds[i,0] = time_points[i]
                follicles_ds[i,1] = self.follicles[indices[i], time_points[i]]
        return follicles_ds
    

    def age_distribution_by_follicles(self,nfollicles):
        #returns the age distribution of individuals with follicle count == nfollicles (or the nearest value<nfollicles)
     

        if self.follicles is None:
            raise ValueError('No follicles were saved or calculated. Run the simulation first.')
        follicles = self.follicles
        ages=[]
        for woman in follicles:
            age =  np.argmin(np.abs(np.array(woman)-nfollicles))
            ages.append(age)
        return ages
    

    

def baysianFollicleDistance(data, sim,log_scale=True, data_times_discretisation = 2):
    """
    Calculate the likelihood of the follicle counts in the data being generated by the follicle counts in the simulation.
    data should be an np array of shape (npoints,2) where data[:,0] is the time and data[:,1] is the follicle count.
    """
    dists = sim.getFoliiclesDists(log_scale=log_scale, interval = data_times_discretisation)
    logps = 0
    for i in range(len(data)):
        t = data[i,0]
        follicle_count = data[i,1]
        if log_scale:
            follicle_count = np.log(follicle_count)
        kde = dists[int(t)]
        logp = np.log(kde.pdf([follicle_count]))
        if(logp==-np.inf):
            print('logp:',logp, 't:', t, 'follicle_count:',follicle_count)
        if np.isnan(logp):
            print('logp is nan: t=',t,'follicle_count=',follicle_count)
            return -np.inf
        logps+=logp
    return logps


def menopauseDistributionMetric(data, sim, n_folllicles_for_meno =1000,debug =True):    
    """
    Calculate the likelihood of the distribution of menopause ages (the age at which the follicle count reaches n_folllicles_for_meno) in the data being generated by the simulation (p(data|sim)).

    Parameters:
    data (Any): The data containing menopause ages. It assumes the data is a 2D numpy array with the first column being the age and the second column being the follicle count.
    sim (SR_Follicles): The simulation object that provides the age distribution by follicles.
    n_folllicles_for_meno (int, optional): The number of follicles at which menopause is considered to occur. Default is 1000.

    Returns:
    float: The log probability of the data given the simulation.
    """
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    ages = sim.age_distribution_by_follicles(n_folllicles_for_meno)
    ages = np.array(ages)
    if debug:
        if np.any(np.isnan(ages)):
            print('ages has nan values')
    ages = ages[~np.isnan(ages)]
    kde = KDEMultivariate(ages, var_type='c', bw='normal_reference')
    cleaned_data = data[data[:,1]==n_folllicles_for_meno]
    data_ages = cleaned_data[:,0]
    logps = np.sum(np.log(kde.pdf(data_ages)))
    return logps
    




#example metric function
def baysianDistance(sr1, sr2, time_range=None, dt =1, debug = False):
    """
    Calculate the likelihood that the death times of sr1 are generated by sr2.
    convention is that sr1 is the data and sr2 is the simulation
    """
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    
    #if number of deathtimes is too small that causes issues and anyways not probable as a legitimate parameter set
    if len(sr2.getDeathTimes()) <= 5:
        return -np.inf
    
    death_times2 = sr2.getDeathTimes()
    events2 = sr2.events
    #if time range is not None, use only deathtimes withtin the time range
    if time_range is not None:
        events2 = events2[(death_times2 >= time_range[0]) & (death_times2 <= time_range[1])]
        death_times2 = death_times2[(death_times2 >= time_range[0]) & (death_times2 <= time_range[1])]
    #check that there are enough events to calculate the likelihood meaningfully
    if np.sum(events2) <= 5:
        return -np.inf
    
    death_times2 = death_times2[events2==1] #only those who died
    #this would be a smooth distribution generatated from sr2(simulation) to sample sr1(data) from
    kde = KDEMultivariate(death_times2, var_type='c', bw='normal_reference')
    
    
    events = sr1.events
    death_times = sr1.getDeathTimes()
    if time_range is not None:
        events = events[(death_times >= time_range[0]) & (death_times <= time_range[1])]
        death_times = death_times[(death_times >= time_range[0]) & (death_times <= time_range[1])]
    died = death_times[events==1]
    censored = death_times[events==0]
    ndied = len(died)
    p_death_before_t_end =np.sum(events2)/len(events2)
    
    
    #this piece of code is to accelerate the loglikelihood calculation
    times = np.linspace(0, max(died)+dt, int(np.ceil(max(died)+dt/dt)+1))
    log_pdt = kde.cdf(times) #the log integral of the probability density function on the time grid
    log_pdt = np.log(log_pdt[1:]-log_pdt[:-1])
    times = times[:-1]

    #if logp has nan values then try again then raise an error to debug
    if np.any(np.isnan(log_pdt)):
        if debug:
            print('log_pdt has nan values')
        return np.NaN

    logcdf = np.log(1-kde.cdf(times)) 
    #for every time in death times, find the nearst index in times and get the logp
    logps = 0
    logps_censored = 0
    for t in died:
        idx = np.argmin(np.abs(times-t))
        logps+=(log_pdt[idx])
        if debug:
            if log_pdt[idx] == -np.inf or np.isnan(log_pdt[idx]):
                print(f'log_pdt[{idx}] is {log_pdt[idx]}')

    for t in censored:
        idx = np.argmin(np.abs(times-t))
        logps_censored+=(logcdf[idx])

    #the liklihod given by the kde is L= p(t_death=x_i)|died before t_end) so to correct we need to multiply by the number of ndied/n
    #in the loglikelihood this gives a term of ndied*np.log(p_death_before_t_end)
    sums = logps+ndied*np.log(p_death_before_t_end) +logps_censored
    #check if sums is nan if so then raise an error to debug.
    if np.isnan(sums):
        print('ndied:',ndied)
        print('p_death_before_t_end:',p_death_before_t_end)
        print('logps:',logps)
        print('censored:',censored)
        print('censored_logLiklihood(censored):',logps_censored)
        raise ValueError('sums is nan')
    if debug:
        print('ndied:',ndied)
        print('p_death_before_t_end:',p_death_before_t_end)
        print('logps:',logps)
        print('censored:',censored)
        print('censored_logLiklihood(censored):',logps_censored)
        print('sums:',sums)
    return sums



def model(theta , n, nsteps, t_end, dataSet, sim=None, metric = 'baysianFollicleDistance', time_range=None, time_step_multiplier = 1,parallel = False, dt=2, set_params=None, kwargs=None):
    """
    The function accepts the parameters of the SR model and returns the KL distance between the two models.
    """
    pv = srmc.parse_theta(theta, set_params)
    eta = pv['eta']
    beta = pv['beta']
    epsilon = pv['epsilon']
    xc = pv['xc']
    external_hazard = pv['external_hazard']
    params = [eta, beta, epsilon, xc]
    initial_follicles_mean = kwargs.get('initial_follicles_mean')
    initial_follicles_std = kwargs.get('initial_follicles_std')
    log_initial_follicles = kwargs.get('log_initial_follicles')
    sim = getSrFollicles(theta,params, n, nsteps, t_end, external_hazard = external_hazard, time_step_multiplier=time_step_multiplier,parallel=parallel, initial_follicles_mean=initial_follicles_mean, initial_follicles_std=initial_follicles_std, log_initial_follicles=log_initial_follicles) if sim is None else sim
   
    if metric == 'baysianFollicleDistance':
        tprob = baysianFollicleDistance(dataSet.follicles, sim, data_times_discretisation= dt)
    elif metric == 'menopauseDistributionMetric':
        tprob = menopauseDistributionMetric(dataSet.follicles, sim)

    return tprob


def getSrFollicles(theta,params =srmc.karin_theta(), npeople =5000, nsteps=5000, t_end=60, external_hazard = np.inf, time_step_multiplier=10,parallel=False, initial_follicles_mean=12.692312588793072, initial_follicles_std=0.441272921587803, log_initial_follicles=True):
    """
    This function returns an instance of the SR_Follicles class.
    """
    lambda0 = theta[0]
    lambda1 = theta[1]
    lambda2 = theta[2]
    eta = params[0]
    beta = params[1]
    kappa = 0.5
    epsilon = params[2]
    xc = params[3]
    sim = SR_Follicles(eta=eta,beta=beta,kappa=kappa,epsilon=epsilon,xc=xc,lambda0=lambda0,lambda1=lambda1,lambda2=lambda2,npeople=npeople,nsteps=nsteps,t_end=t_end,external_hazard=external_hazard,time_step_multiplier=time_step_multiplier,parallel=parallel,initial_follicles_mean=initial_follicles_mean,initial_follicles_std=initial_follicles_std,log_initial_follicles=log_initial_follicles)
    return sim



#method without parallelization (for cluster usage)
@jit(nopython=jit_nopython)
def death_times_accelerator(s,dt,t,eta,beta,kappa,epsilon,xc,lambda0, lambda1, lambda2, t_end, sdt,npeople,external_hazard = np.inf, time_step_multiplier = 1, mean2 = 12.692312588793072, std2 = 0.441272921587803, lognormal = True):
    death_times = []
    events = []
    follicles = []
    for i in range(npeople):
        x=0
        j=0
        normal_data = np.random.normal(loc = mean2,scale = std2)
        N = np.ones(t_end)
        N = N*(-np.inf)
        if lognormal:
            N[0] = np.exp(normal_data)
        else:
            N[0] = normal_data
        nf = N[0] #initial number of follicles
        i_nf = 1
        ndt = dt/time_step_multiplier
        nsdt = sdt/np.sqrt(time_step_multiplier)
        chance_to_die_externally = np.exp(-external_hazard)*ndt
        while j in range(s-1) and x<xc:
            for i in range(time_step_multiplier):
                noise = np.sqrt(2*epsilon)*np.random.normal(loc = 0,scale = 1)
                x = x+ndt*(eta*(t[j]+i*ndt)-beta*x/(x+kappa))+noise*nsdt
                x = np.maximum(x, 0)
                nf = nf-(lambda0+lambda1*x+lambda2*x**2)*nf*ndt
                if np.abs(t[j]-i_nf)<dt:
                    N[i_nf] = nf
                    i_nf+=1
                if np.random.uniform(0,1)<chance_to_die_externally:
                    x = xc
                if x>=xc:
                    break
            j+=1
        follicles.append(N)
        if x>=xc:
            death_times.append(j*dt)
            events.append(1)
        else:
            death_times.append(j*dt)
            events.append(0)

    return death_times, events, follicles

##method with parallelization (run on your computer), I updated but didn't test, ypou can try it and see if it works (use parallel =True in the class)
def death_times_accelerator2(s,dt,t,eta,beta,kappa,epsilon,xc,lambda0, lambda1, lambda2, t_end, sdt,npeople,external_hazard = np.inf, time_step_multiplier = 1, mean2 = 12.692312588793072, std2 = 0.441272921587803, lognormal = True):
    @jit(nopython=jit_nopython)
    def calculate_death_times(npeople, s, dt, t, eta, beta, kappa, epsilon, xc, lambda0, lambda1, lambda2, t_end, sdt,external_hazard = np.inf, time_step_multiplier = 1, mean2 = 12.692312588793072, std2 = 0.441272921587803, lognormal = True):
        death_times = []
        events = []
        follicles = []
        for i in range(npeople):
            died = False
            x=0
            j=0
            normal_data = np.random.normal(loc = mean2,scale = std2)
            N = np.ones(t_end)
            N = N*(-np.inf)
            if lognormal:
                N[0] = np.exp(normal_data)
            else:
                N[0] = normal_data
            nf = N[0] #initial number of follicles
            i_nf = 1
            ndt = dt/time_step_multiplier
            nsdt = np.sqrt(ndt)
            chance_to_die_externally = np.exp(-external_hazard)*ndt
            while j in range(s-1) and x<xc:
                for i in range(time_step_multiplier):
                    noise = np.sqrt(2*epsilon)*np.random.normal(loc = 0,scale = 1)
                    x = x+ndt*(eta*(t[j]+i*ndt)-beta*x/(x+kappa))+noise*nsdt
                    x = np.maximum(x, 0)
                    nf = nf-(lambda0+lambda1*x+lambda2*x**2)*nf*ndt
                    if np.abs(t[j]-i_nf)<dt:
                        N[i_nf] = nf
                        i_nf+=1
                    if np.random.uniform(0,1)<chance_to_die_externally:
                        x = xc
                    if x>=xc:
                        died =True
                j+=1
            follicles.append(N)
            if died:
                death_times.append(j*dt)
                events.append(1)
            else:
                death_times.append(j*dt)
                events.append(0)

        return death_times, events, follicles

    n_jobs = os.cpu_count()
    npeople_per_job = npeople // n_jobs
    results = Parallel(n_jobs=n_jobs)(delayed(calculate_death_times)(
        npeople_per_job, s, dt, t, eta, beta, kappa, epsilon, xc, lambda0, lambda1, lambda2, t_end, sdt, external_hazard, time_step_multiplier, mean2, std2, lognormal
    ) for _ in range(n_jobs))
    death_times = [dt for sublist in results for dt in sublist[0]]
    events = [event for sublist in results for event in sublist[1]]
    follicles = [follicle for sublist in results for follicle in sublist[2]]
    return death_times, events, follicles