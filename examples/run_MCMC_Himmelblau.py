import sys,os
import math
import numpy as np
import scipy
import time
from random import randrange
import Algo_1

# Himmelblau settings
def log_likelihood(x,dim):
        y = np.power(np.power(x[0], 2) + x[1] - 11, 2) + np.power(x[0] + np.power(x[1], 2) - 7, 2)
        likelihood = np.exp(-y)
        return likelihood

outdir = 'Himmelblau_2D'
dim = 2
low_bound = -5*np.ones(dim)
high_bound = 5*np.ones(dim)
sigma = 0.15
retrains = 100
samples_per_retrain = 100
diffusion_prob = 0.83

# Himmel initial modes
initial_sample_size = 2000
maxima = np.array([[3,2],[-2.81,3.13],[-3.78,-3.28],[3.58,-1.85]])
initial_samples = []
for maximum in maxima:
    means = maximum
    cov = [[0.1,0],[0,0.1]]
    initial_samples.extend(np.random.multivariate_normal(means,cov,
    initial_sample_size//len(maxima)))
initial_samples = np.array(initial_samples)

algo1 = Algo_1.MH_Diffusion(log_likelihood,dim,low_bound,high_bound,
initial_samples,retrains,samples_per_retrain,outdir = outdir,
sigma = sigma,diffusion_prob = diffusion_prob)
samples,diffusion_samples,MH_samples,diffusion_rate = algo1.run()

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

isExist = os.path.exists(outdir + '/' + 'diffusion_acceptance.dat')
if isExist:
    os.remove(outdir + '/' + 'diffusion_acceptance.dat')
np.savetxt(outdir + '/' + 'diffusion_acceptance.dat',diffusion_rate)
