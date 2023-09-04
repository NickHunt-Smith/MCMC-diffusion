import sys,os
import math
import numpy as np
import scipy
import time
from random import randrange
import Algo_1

def log_likelihood(x,dim):
    mean_1 = [8,3,0,0,0,0,0,0,0,0]
    std_1 = 0.6666
    mean_2 = [-2,3,0,0,0,0,0,0,0,0]
    std_2 = 0.3333
    gauss_1 = scipy.stats.multivariate_normal.pdf(x,mean =
    mean_1,cov = [[0.6666,0,0,0,0,0,0,0,0,0],[0,0.6666,0,0,0,0,0,0,0,0],[0,0,0.6666,0,0,0,0,0,0,0],[0,0,0,0.6666,0,0,0,0,0,0],[0,0,0,0,0.6666,0,0,0,0,0],[0,0,0,0,0,0.6666,0,0,0,0],[0,0,0,0,0,0,0.6666,0,0,0],[0,0,0,0,0,0,0,0.6666,0,0],[0,0,0,0,0,0,0,0,0.6666,0],[0,0,0,0,0,0,0,0,0,0.6666]])
    gauss_2 = scipy.stats.multivariate_normal.pdf(x,mean =
    mean_2*np.ones(dim),cov = [[0.3333,0,0,0,0,0,0,0,0,0],[0,0.3333,0,0,0,0,0,0,0,0],[0,0,0.3333,0,0,0,0,0,0,0],[0,0,0,0.3333,0,0,0,0,0,0],[0,0,0,0,0.3333,0,0,0,0,0],[0,0,0,0,0,0.3333,0,0,0,0],[0,0,0,0,0,0,0.3333,0,0,0],[0,0,0,0,0,0,0,0.3333,0,0],[0,0,0,0,0,0,0,0,0.3333,0],[0,0,0,0,0,0,0,0,0,0.3333]])
    likelihood = gauss_1 + gauss_2
    return likelihood

outdir = 'Gaussian_10D'
dim = 10
low_bound = -5*np.ones(dim)
high_bound = 10*np.ones(dim)
sigma = 0.5
retrains = 100
samples_per_retrain = 100
diffusion_prob = 0.2
plot_initial = False

# Generate pure Metropolis-Hastings comparison samples
sigma_pureMH = 0.5
comparison_sample_size = 10000
pureMH = Algo_1.PureMH(log_likelihood,comparison_sample_size,dim,
low_bound,high_bound,sigma_pureMH = sigma_pureMH)
comparison_samples = np.array(pureMH.run())
isExist = os.path.exists(outdir)
if not isExist:
    os.mkdir(outdir)
np.savetxt(outdir + '/' + 'comparison_samples.dat',comparison_samples)

# Gaussian initial samples, we pretend we don't know the relative weights of the modes
initial_sample_size = 100
maxima = np.array([[8,3,0,0,0,0,0,0,0,0],[-2,3,0,0,0,0,0,0,0,0]])
covs = [[[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],[[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]]
initial_samples = []
for i in range(0,len(maxima)):
    means = maxima[i]
    cov = covs[i]
    initial_samples.extend(np.random.multivariate_normal(means,cov,
    initial_sample_size//len(maxima)))
initial_samples = np.array(initial_samples)

algo1 = Algo_1.MH_Diffusion(log_likelihood,dim,low_bound,high_bound,
initial_samples,retrains,samples_per_retrain,outdir = outdir,sigma = sigma,diffusion_prob = diffusion_prob,plot_initial = plot_initial)
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
