import sys,os
import math
import numpy as np
import scipy
import time
from random import randrange
import Algo_1

# Rosenbrock Settings
def log_likelihood(x,dim):
    y = 0
    for k in range(dim-1):
        y += (100 * np.power(x[k + 1] - np.power(x[k], 2), 2) +
              np.power(1 - x[k], 2))
    likelihood = np.exp(-y)
    return likelihood

outdir = 'Rosenbrock_4D'
dim = 4
low_bound = -3*np.ones(dim)
high_bound = 3*np.ones(dim)
sigma = 0.3
noise_width = 0.05
initial_sample_size = 10000
retrains = 100
samples_per_retrain = 2000

# Rosenbrock Initial samples
initial_samples = []
for dim_iter in range(0,dim):
    initial_samples.append(np.random.uniform(low = low_bound[dim_iter],high = high_bound[dim_iter],size = initial_sample_size))
initial_samples = np.array(initial_samples).T

# Generate pure Metropolis-Hastings comparison samples
sigma_pureMH = 0.3
comparison_sample_size = 200000
pureMH = Algo_1.PureMH(log_likelihood,comparison_sample_size,dim,low_bound,high_bound,sigma_pureMH = sigma_pureMH)
comparison_samples = np.array(pureMH.run())
isExist = os.path.exists(outdir)
if not isExist:
    os.mkdir(outdir)
np.savetxt(outdir + '/' + 'comparison_samples.dat',comparison_samples)

algo1 = Algo_1.MH_Diffusion(log_likelihood,dim,low_bound,high_bound,initial_samples,retrains,samples_per_retrain,outdir = outdir,sigma = sigma, noise_width = noise_width)
samples,diffusion_samples,MH_samples = algo1.run()

isExist = os.path.exists(outdir + '/' + 'samples.dat')
if isExist:
    os.remove(outdir + '/' + 'samples.dat')
np.savetxt(outdir + '/' + 'samples.dat',samples)

isExist = os.path.exists(outdir + '/' + 'diffusion_samples.dat')
if isExist:
    os.remove(outdir + '/' + 'diffusion_samples.dat')
np.savetxt(outdir + '/' + 'diffusion_samples.dat',diffusion_samples)

isExist = os.path.exists(outdir + '/' + 'MH_samples.dat')
if isExist:
    os.remove(outdir + '/' + 'MH_samples.dat')
np.savetxt(outdir + '/' + 'MH_samples.dat',MH_samples)
