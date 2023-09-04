import sys,os
import numpy as np
from scipy import stats
import diffusion
import time
from random import randrange
import pygtc

def lprint(msg):
    sys.stdout.write('\r')
    sys.stdout.write(msg)
    sys.stdout.flush()

class MH_Diffusion:
    def __init__(self,log_likelihood,dim,low_bound,high_bound,initial_samples,
    retrains,samples_per_retrain,outdir = 'chain',nsteps = 20,sigma = 0.3,
    diffusion_prob = 0.5,bins = 20,noise_width = 0.05,beta_1 = 0.1,beta_2 = 0.3,plot_initial = True):
        self.log_likelihood = log_likelihood
        self.dim = dim
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.initial_samples = initial_samples
        self.retrains = retrains
        self.samples_per_retrain = samples_per_retrain
        self.outdir = outdir
        self.nsteps = nsteps
        self.sigma = sigma
        self.diffusion_prob = diffusion_prob
        self.bins = bins
        self.noise_width = noise_width
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.plot_initial = plot_initial


    def run(self):
        print('Diffusion MCMC Chain Started')

        isExist = os.path.exists(self.outdir)
        if not isExist:
            os.mkdir(self.outdir)

        training_samples = self.initial_samples
        initial_sample_size = len(training_samples)
        desired_sample_size = initial_sample_size
        var_guess = []
        for _ in range(0,self.nsteps):
            var_guess.append(1)
        var_guess = np.array(var_guess)

        # Train diffusion model on initial seeded samples
        model = diffusion.DiffusionModel(self.dim,self.nsteps,self.noise_width,
        initial_sample_size,desired_sample_size,training_samples,var_guess,self.beta_1,self.beta_2)
        diffusion_samples,vars = model.fit()

        if self.plot_initial:
            chainLabels = ['Diffusion Samples', 'Starting Distribution']
            Ranges = []
            for dim_iter in range(0,self.dim):
                Ranges.append([self.low_bound[dim_iter],
                self.high_bound[dim_iter]])
            GTC = pygtc.plotGTC(chains=[np.array(diffusion_samples),
            np.array(training_samples)],nContourLevels = 3,
            chainLabels = chainLabels,paramRanges = Ranges,do1dPlots=True,
            nBins  = 20,figureSize = 10,
            customLegendFont = {'family':'Times New Roman', 'size':20},
            customTickFont = {'family':'Times New Roman', 'size':10})
            ax = GTC.axes

            GTC.savefig(self.outdir + '/' + 'diffusion_check.pdf')

        samples_final = self.initial_samples
        theta = samples_final[np.random.randint(0,len(samples_final))]
        accepted_diffusion = []
        accepted_MH = []
        total_time = 0
        diffusion_rate = []

        for retrain_iter in range(self.retrains):

            start = time.time()

            H = []
            edges = []
            for dim_iter in range(0,self.dim):
                H_temp,edges_temp = np.histogram(diffusion_samples[:,dim_iter],
                bins = self.bins)
                H.append(H_temp/np.sum(H_temp))
                edges.append(edges_temp)

            naccepted_diffusion = 0
            nattempted_diffusion = 0
            naccepted = 0
            nattempted = 0
            for i in range(self.samples_per_retrain):
                rand = np.random.uniform()
                # Diffusion as proposal some of the time
                if rand < self.diffusion_prob:
                    nattempted_diffusion +=1
                    rand_pick = randrange(len(diffusion_samples))
                    theta_prime = diffusion_samples[rand_pick]

                    edge_loc = []
                    for dim_iter in range(0,self.dim):
                        for edge_iter in range(0,len(edges[dim_iter])):
                            if theta_prime[dim_iter] >= edges[dim_iter][len(edges[dim_iter])-1]:
                                edge_loc.append(len(edges[dim_iter])-2)
                                break
                            elif theta_prime[dim_iter] < edges[dim_iter][edge_iter]:
                                edge_loc.append(edge_iter-1)
                                break

                    Q_prime = 1
                    for dim_iter in range(0,self.dim):
                        Q_prime = Q_prime*H[dim_iter][edge_loc[dim_iter]]
                    if Q_prime == float(0):
                        Q_prime = 0.000001

                    edge_loc = []
                    for dim_iter in range(0,self.dim):
                        for edge_iter in range(0,len(edges[dim_iter])):
                            if theta[dim_iter] >= edges[dim_iter][len(edges[dim_iter])-1]:
                                edge_loc.append(len(edges[dim_iter])-2)
                                break
                            elif theta[dim_iter] < edges[dim_iter][edge_iter]:
                                edge_loc.append(edge_iter-1)
                                break

                    Q = 1
                    for dim_iter in range(0,self.dim):
                        Q = Q*H[dim_iter][edge_loc[dim_iter]]
                    if Q == float(0):
                        Q = 0.000001

                    Q_ratio = Q/Q_prime


                    for j in range(self.dim):
                        while theta_prime[j] < self.low_bound[j] or theta_prime[j] > self.high_bound[j]:
                            theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()
                    L_ratio = self.log_likelihood(theta_prime,self.dim)/self.log_likelihood(theta,self.dim)
                    prob_accept = L_ratio*Q_ratio
                    a = min(1, prob_accept)
                    u = np.random.uniform()
                    if u < a:
                        naccepted_diffusion +=1
                        theta = theta_prime
                        accepted_diffusion.append(theta_prime)
                # M-H the rest of the time
                else:
                    nattempted +=1
                    theta_prime = np.zeros(self.dim)
                    for j in range(self.dim):
                        theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()
                        while theta_prime[j] < self.low_bound[j] or theta_prime[j] > self.high_bound[j]:
                            theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()
                    theta_prime = np.array(theta_prime)
                    a = min(1, self.log_likelihood(theta_prime,self.dim)/self.log_likelihood(theta,self.dim))
                    u = np.random.uniform()
                    if u < a:
                        naccepted +=1
                        theta = theta_prime
                        accepted_MH.append(theta_prime)
                samples_final = np.vstack((samples_final,theta))

            # Retrain Diffusion
            initial_sample_size = len(samples_final)
            desired_sample_size = 10*initial_sample_size
            training_samples = samples_final
            var_guess = vars
            model = diffusion.DiffusionModel(self.dim,self.nsteps,
            self.noise_width,initial_sample_size,desired_sample_size,
            training_samples,var_guess,self.beta_1,self.beta_2)
            diffusion_samples,vars = model.fit()
            end = time.time()

            total_time += end-start

            print('Number of retrains = ' + str(retrain_iter+1) + '/' +
            str(self.retrains) + '\n' + 'Diffusion acceptance efficiency = ' +
            str(naccepted_diffusion/nattempted_diffusion) + '\n' +
            'Metropolis acceptance efficiency = ' + str(naccepted/nattempted) +
            '\n' + 'Previous Retrain Time = ' + str(np.round(end-start)) +
            ' seconds ' + '\n' + 'Total Retrain Time = ' +
            str(np.round(total_time)) + ' seconds ')

            if not retrain_iter == self.retrains-1:
                for _ in range(0,5):
                    UP = '\033[1A'
                    CLEAR = '\x1b[2K'
                    print(UP, end=CLEAR)

            diffusion_rate.append(naccepted_diffusion/nattempted_diffusion)
        return samples_final,accepted_diffusion,accepted_MH,diffusion_rate


class PureMH:

    def __init__(self,log_likelihood,nsamples,dim,low_bound,high_bound,sigma_pureMH = 0.3):
        self.log_likelihood = log_likelihood
        self.nsamples = nsamples
        self.dim = dim
        self.sigma = sigma_pureMH
        self.low_bound = low_bound
        self.high_bound = high_bound

    def run(self):
        theta = np.zeros(self.dim)
        theta_prime = np.zeros(self.dim)
        naccepted = 0
        samples = []

        for i in range(self.nsamples):
            lprint('Pure MH samples = ' + str(i))
            theta_prime = np.zeros(self.dim)
            for j in range(self.dim):
                theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs() # proposal sample, taken from gaussian centred on current sample with std = sigma
                while theta_prime[j] < self.low_bound[j] or theta_prime[j] > self.high_bound[j]:
                    theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()
            a = min(1, self.log_likelihood(theta_prime,self.dim)/self.log_likelihood(theta,self.dim))
            u = np.random.uniform()
            if u < a:
                naccepted +=1
                theta = theta_prime
            samples.append(theta)
        print(' Pure MH Acceptance efficiency = ',naccepted/self.nsamples)
        return samples
