import sys,os
import math
import numpy as np
import scipy
import time
from random import randrange
import Algo_1

# Double Gaussian settings
def log_likelihood(x,dim):
    mean_1 = 2
    std_1 = 1
    mean_2 = 7
    std_2 = 0.5
    gauss_1 = scipy.stats.multivariate_normal.pdf(x,mean =
    [mean_1,mean_1],cov = [std_1,std_1])
    gauss_2 = scipy.stats.multivariate_normal.pdf(x,mean =
    [mean_2,mean_2],cov = [std_2,std_2])
    likelihood = gauss_1 + gauss_2
    return likelihood

outdir = 'Gaussian_2D'
dim = 2
low_bound = np.zeros(dim)
high_bound = 10*np.ones(dim)
sigma = 2
retrains = 100
samples_per_retrain = 500

# Generate pure Metropolis-Hastings comparison samples
sigma_pureMH = 2
comparison_sample_size = 50000
pureMH = Algo_1.PureMH(log_likelihood,comparison_sample_size,dim,
low_bound,high_bound,sigma_pureMH = sigma_pureMH)
comparison_samples = np.array(pureMH.run())
isExist = os.path.exists(outdir)
if not isExist:
    os.mkdir(outdir)
np.savetxt(outdir + '/' + 'comparison_samples.dat',comparison_samples)

# Can then use some of these pure MH samples as initial samples for diffusion model
comparison_samples = np.loadtxt(outdir + '/' + 'comparison_samples.dat')
initial_samples = comparison_samples[:5000]

algo1 = Algo_1.MH_Diffusion(log_likelihood,dim,low_bound,high_bound,
initial_samples,retrains,samples_per_retrain,outdir = outdir,sigma = sigma)
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
