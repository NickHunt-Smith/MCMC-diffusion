import sys,os
import math
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import stats
import corner
from scipy.optimize  import minimize
import time
from random import randrange
import inspect
import time
import Algo_1

# PDF setup
num_points = 20
seed = 10
def get_u(x,a,b):
    return x**a*(1-x)**b

def get_d(x,a,b):
    return 0.1*x**a*(1-x)**b

def get_sigma1(x,p):
    u=get_u(x,p[0],p[1])
    d=get_d(x,p[2],p[3])
    return 4*u+d


def get_sigma2(x,p):
    u=get_u(x,p[0],p[1])
    d=get_d(x,p[2],p[3])
    return 4*d+u

np.random.seed(seed)
true_params=[0.5,2.5,0.1,3.0]

X1=np.linspace(0.1,0.9,num_points)
sigma1T=get_sigma1(X1,true_params)
dsigma1=0.1*sigma1T
sigma1=sigma1T+dsigma1*np.random.randn(len(sigma1T))

X2=np.linspace(0.1,0.9,num_points)
sigma2T=get_sigma2(X2,true_params)
dsigma2=0.1*sigma2T
sigma2=sigma2T+dsigma2*np.random.randn(len(sigma2T))

def log_likelihood(p,dim):
    res1=(sigma1-get_sigma1(X1,p))/dsigma1
    res2=(sigma2-get_sigma2(X2,p))/dsigma2
    res = np.append(res1,res2)
    chi2_temp = np.sum(res**2)
    likelihood = np.exp(-chi2_temp)
    return likelihood

# PDF settings
outdir = 'PDF_4D'
dim = 4
low_bound = [-1,0,-1,0]
high_bound = [1,5,1,5]
sigma = 0.02
noise_width = 0.1
retrains = 100
samples_per_retrain = 1000
plot_initial = False

# PDF initial mode
initial_sample_size = 1000
maxima = np.array([[0.5,2.5,0.1,3]])
initial_samples = []
for maximum in maxima:
    means = maximum
    cov = [[0.1,0,0,0],[0,0.1,0,0],[0,0,0.1,0],[0,0,0,0.1]]
    initial_samples.extend(np.random.multivariate_normal(means,cov,initial_sample_size//len(maxima)))
initial_samples = np.array(initial_samples)

# Generate pure Metropolis-Hastings comparison samples
sigma_pureMH = 0.1
comparison_sample_size = 100000
pureMH = Algo_1.PureMH(log_likelihood,comparison_sample_size,dim,
low_bound,high_bound,sigma_pureMH = sigma_pureMH)
comparison_samples = np.array(pureMH.run())
isExist = os.path.exists(outdir)
if not isExist:
    os.mkdir(outdir)
np.savetxt(outdir + '/' + 'comparison_samples.dat',comparison_samples)

algo1 = Algo_1.MH_Diffusion(log_likelihood,dim,low_bound,high_bound,initial_samples,retrains,samples_per_retrain,outdir = outdir,sigma = sigma, noise_width = noise_width, plot_initial = plot_initial)
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
