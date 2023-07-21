import sys,os
from subprocess import Popen, PIPE
import string

def lprint(msg):
    sys.stdout.write('\r')
    sys.stdout.write(msg)
    sys.stdout.flush()

import numpy as np
import pandas as pd
import random
from random import randrange
import math
import pygmo as pg
import corner
from scipy import stats
import scipy

import latex
import matplotlib
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import pylab as py
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text',usetex=True)

from scipy import optimize
from scipy.optimize  import minimize,leastsq
from scipy.optimize import least_squares

outdir = 'PDF_4D'
num_points = 20
num_reps = 10000
dim = 4

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

np.random.seed(10)
true_params=[0.5,2.5,0.1,3.0]

X1=np.linspace(0.1,0.9,num_points)
sigma1T=get_sigma1(X1,true_params)
dsigma1=0.1*sigma1T
sigma1=sigma1T+dsigma1*np.random.randn(len(sigma1T))

X2=np.linspace(0.1,0.9,num_points)
sigma2T=get_sigma2(X2,true_params)
dsigma2=0.1*sigma2T
sigma2=sigma2T+dsigma2*np.random.randn(len(sigma2T))

pmin=np.array([-1,0,-1,0])
pmax=np.array([+1,5,+1,5])

# Data Resampling for approximate true distribution
replicas_dr=[]
for irep in range(num_reps):
    if irep%10==0: lprint('%d'%(irep))
    ksigma1=sigma1+(dsigma1)*np.random.randn(len(sigma1))
    ksigma2=sigma2+(dsigma2)*np.random.randn(len(sigma2))

    def get_residuals(p):
        res1=(ksigma1-get_sigma1(X1,p))/dsigma1
        res2=(ksigma2-get_sigma2(X2,p))/dsigma2
        res = np.append(res1,res2)
        return res

    fit=least_squares(get_residuals,pmin+(pmax-pmin)*np.random.uniform(0,1,4),bounds=np.array([pmin,pmax]))
    replicas_dr.append(fit.x)

    data_file_name = outdir + '/replicas.dat'
    isExist = os.path.exists(data_file_name)
    if isExist:
        os.remove(data_file_name)
    F = open(data_file_name,'a')
    data_mean_string = ''
    for i in range(len(replicas_dr)):
        for j in range(len(replicas_dr[i])):
            data_mean_string = data_mean_string + str(replicas_dr[i][j]) + ' '
        data_mean_string = data_mean_string + '\n'
    F.write(data_mean_string)
    F.close()

dim = 4
samples_total = 10100
samples_burn_in = 100

algo1_samples = np.loadtxt(outdir + '/samples.dat')
comparison_samples = np.loadtxt(outdir + '/comparison_samples.dat')
replicas_plot = np.loadtxt(outdir + '/replicas.dat')

def gen_inference(X1,X2,X,replicas):
    sigma1,sigma2=[],[]
    for par in replicas:
        sigma1.append(get_sigma1(X1,par))
        sigma2.append(get_sigma2(X2,par))

    u,d=[],[]
    for par in replicas:
        u.append(get_u(X,par[0],par[1]))
        d.append(get_d(X,par[2],par[3]))

    data={}
    data['sigma1']=np.mean(sigma1,axis=0)
    data['sigma2']=np.mean(sigma2,axis=0)
    data['dsigma1']=np.std(sigma1,axis=0)
    data['dsigma2']=np.std(sigma2,axis=0)

    data['u']=np.mean(u,axis=0)
    data['d']=np.mean(d,axis=0)
    data['du']=np.std(u,axis=0)
    data['dd']=np.std(d,axis=0)
    return data

pp = PdfPages(outdir + '/PDFs.pdf')
ncols = 2
nrows = 3
fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*7,nrows*5))
plt.subplots_adjust(wspace = 0.25)
plt.subplots_adjust(hspace = 0.25)

X=np.linspace(0.1,0.9,100)
data_dr=gen_inference(X,X,X,replicas_plot)
data_diffusion = gen_inference(X,X,X,algo1_samples[samples_burn_in:samples_total])
data_MH=gen_inference(X,X,X,comparison_samples[samples_burn_in:samples_total])

edgecolors = ['r','dodgerblue','orange']

norm1=get_sigma1(X,true_params)
norm2=get_sigma2(X,true_params)
normu=get_u(X,true_params[0],true_params[1])
normd=get_d(X,true_params[2],true_params[3])
ax[0][0].errorbar(X1,sigma1,dsigma1,ecolor = 'k',color = 'k',fmt='.',markersize=10,label=r'\boldmath{$ \sigma^{\rm dat}_1$}')
ax[0][0].errorbar(X2,sigma2,dsigma2,ecolor = 'k',color = 'k',fmt='.',markersize=10,mfc='white',label=r'\boldmath{$ \sigma^{\rm dat}_2$}')
ax[0][0].plot(X,get_sigma1(X,true_params),'b',lw=2,label=r'\boldmath{$\sigma_1$}')
ax[0][0].plot(X,get_sigma2(X,true_params),'forestgreen',lw=2,label = r'\boldmath{$\sigma_2$}')
ax[0][0].fill_between(X,data_dr['sigma1']-data_dr['dsigma1'],data_dr['sigma1']+data_dr['dsigma1']
                ,facecolor='none',hatch = '/',edgecolor=edgecolors[0])
ax[0][0].fill_between(X,data_diffusion['sigma1']-data_diffusion['dsigma1'],data_diffusion['sigma1']+data_diffusion['dsigma1']
                ,facecolor='none',hatch = '..',edgecolor=edgecolors[1])
ax[0][0].fill_between(X,data_MH['sigma1']-data_MH['dsigma1'],data_MH['sigma1']+data_MH['dsigma1']
                ,facecolor='none',hatch = '--',edgecolor=edgecolors[2])
ax[0][0].fill_between(X,data_dr['sigma2']-data_dr['dsigma2'],data_dr['sigma2']+data_dr['dsigma2']
                ,facecolor='none',hatch = '/',edgecolor=edgecolors[0])
ax[0][0].fill_between(X,data_diffusion['sigma2']-data_diffusion['dsigma2'],data_diffusion['sigma2']+data_diffusion['dsigma2']
                ,facecolor='none',hatch = '..',edgecolor=edgecolors[1])
ax[0][0].fill_between(X,data_MH['sigma2']-data_MH['dsigma2'],data_MH['sigma2']+data_MH['dsigma2']
                ,facecolor='none',hatch = '--',edgecolor=edgecolors[2])

ax[0][0].set_xlabel(r'\boldmath{$x$}',fontsize = 30)
ax[0][0].set_ylabel(r'\boldmath{$\sigma(x)$}',fontsize = 30,labelpad=10)
ax[0][0].set_ylim(0,1.4)
ax[0][0].set_xlim(0.07,0.9)
ax[0][0].tick_params(axis='both', which='both', labelsize=20,direction='in', length=6)
ax[0][0].legend(fontsize = 18, frameon=0,handlelength=1.5,handletextpad=0.6)

ax[1][0].plot(X,get_sigma1(X,true_params)/norm1,'b',lw = 5)
ax[1][0].fill_between(X,(data_dr['sigma1']-data_dr['dsigma1'])/norm1
                 ,(data_dr['sigma1']+data_dr['dsigma1'])/norm1
                 ,facecolor='none',hatch = '/',edgecolor=edgecolors[0])
ax[1][0].fill_between(X,(data_diffusion['sigma1']-data_diffusion['dsigma1'])/norm1
                 ,(data_diffusion['sigma1']+data_diffusion['dsigma1'])/norm1
                 ,facecolor='none',hatch = '..',edgecolor=edgecolors[1])
ax[1][0].fill_between(X,(data_MH['sigma1']-data_MH['dsigma1'])/norm1
                 ,(data_MH['sigma1']+data_MH['dsigma1'])/norm1
                 ,facecolor='none',hatch = '--',edgecolor=edgecolors[2])
ax[1][0].set_xlim(0.07,0.9)
ax[1][0].set_ylim([0.9,1.1])
ax[1][0].set_xlabel(r'\boldmath{$x$}',size=30)
ax[1][0].set_ylabel(r'\boldmath{${\rm ratio~to}~\sigma_1$}',size=30,labelpad=10)
ax[1][0].tick_params(axis='both', which='both', labelsize=20,direction='in', length=6)

ax[2][0].plot(X,get_sigma2(X,true_params)/norm2,'forestgreen',lw = 5)
ax[2][0].fill_between(X,(data_dr['sigma2']-data_dr['dsigma2'])/norm2
                 ,(data_dr['sigma2']+data_dr['dsigma2'])/norm2
                 ,facecolor='none',hatch = '/',edgecolor=edgecolors[0])
ax[2][0].fill_between(X,(data_diffusion['sigma2']-data_diffusion['dsigma2'])/norm2
                 ,(data_diffusion['sigma2']+data_diffusion['dsigma2'])/norm2
                 ,facecolor='none',hatch = '..',edgecolor=edgecolors[1])
ax[2][0].fill_between(X,(data_MH['sigma2']-data_MH['dsigma2'])/norm2
                 ,(data_MH['sigma2']+data_MH['dsigma2'])/norm2
                 ,facecolor='none',hatch = '--',edgecolor=edgecolors[2])
ax[2][0].set_xlim(0.07,0.9)
ax[2][0].set_ylim([0.9,1.1])
ax[2][0].set_xlabel(r'\boldmath{$x$}',size=30)
ax[2][0].set_ylabel(r'\boldmath{${\rm ratio~to}~\sigma_2$}',size=30,labelpad=10)
ax[2][0].tick_params(axis='both', which='both',labelsize=20,direction='in', length=6)

ax[0][1].plot(X,get_u(X,0.5,2.5),label=r'\boldmath{$q_1$}',lw=2,color = 'b')
ax[0][1].plot(X,get_d(X,0.1,3.0),label=r'\boldmath{$q_2$}',lw=2,color = 'forestgreen')
ax[0][1].fill_between(X,data_dr['u']-data_dr['du'],data_dr['u']+data_dr['du']
                ,facecolor='none',hatch = '/',edgecolor=edgecolors[0],label=r'\rm \bf Many Samples')
ax[0][1].fill_between(X,data_diffusion['u']-data_diffusion['du'],data_diffusion['u']+data_diffusion['du']
                ,facecolor='none',hatch = '..',edgecolor=edgecolors[1],label=r'\rm \bf Diffusion + MH')
ax[0][1].fill_between(X,data_MH['u']-data_MH['du'],data_MH['u']+data_MH['du']
                ,facecolor='none',hatch = '--',edgecolor=edgecolors[2],label=r'\rm \bf Pure MH')
ax[0][1].fill_between(X,data_dr['d']-data_dr['dd'],data_dr['d']+data_dr['dd']
                ,facecolor='none',hatch = '/',edgecolor=edgecolors[0])
ax[0][1].fill_between(X,data_diffusion['d']-data_diffusion['dd'],data_diffusion['d']+data_diffusion['dd']
                ,facecolor='none',hatch = '..',edgecolor=edgecolors[1])
ax[0][1].fill_between(X,data_MH['d']-data_MH['dd'],data_MH['d']+data_MH['dd']
                ,facecolor='none',hatch = '--',edgecolor=edgecolors[2])
ax[0][1].set_xlabel(r'\boldmath{$x$}',fontsize = 30)
ax[0][1].set_ylabel(r'\boldmath{$f(x)$}',fontsize = 30,labelpad=10)
ax[0][1].set_xlim(0.07,0.9)
ax[0][1].set_ylim(0,0.35)
ax[0][1].tick_params(axis='both', which='both', labelsize=20,direction='in', length=6)
ax[0][1].legend(fontsize = 15, frameon=0,handlelength=1.5,handletextpad=0.6)

ax[1][1].plot(X,get_u(X,true_params[0],true_params[1])/normu,'b',lw = 5)
ax[1][1].fill_between(X,(data_dr['u']-data_dr['du'])/normu
                 ,(data_dr['u']+data_dr['du'])/normu
                 ,facecolor='none',hatch = '/',edgecolor=edgecolors[0])
ax[1][1].fill_between(X,(data_diffusion['u']-data_diffusion['du'])/normu
                 ,(data_diffusion['u']+data_diffusion['du'])/normu
                 ,facecolor='none',hatch = '..',edgecolor=edgecolors[1])
ax[1][1].fill_between(X,(data_MH['u']-data_MH['du'])/normu
                 ,(data_MH['u']+data_MH['du'])/normu
                 ,facecolor='none',hatch = '--',edgecolor=edgecolors[2])
ax[1][1].set_xlim(0.07,0.9)
ax[1][1].set_ylim([0.9,1.1])
ax[1][1].set_xlabel(r'\boldmath{$x$}',size=30)
ax[1][1].set_ylabel(r'\boldmath{${\rm ratio~to}~q_1$}',size=30,labelpad=10)
ax[1][1].tick_params(axis='both', which='both', labelsize=20,direction='in', length=6)

ax[2][1].plot(X,get_d(X,true_params[2],true_params[3])/normd,'forestgreen',lw = 5)
ax[2][1].fill_between(X,((data_dr['d']-data_dr['dd'])/normd)
                 ,((data_dr['d']+data_dr['dd'])/normd)
                 ,facecolor='none',hatch = '/',edgecolor=edgecolors[0])
ax[2][1].fill_between(X,((data_diffusion['d']-data_diffusion['dd'])/normd)
                 ,((data_diffusion['d']+data_diffusion['dd'])/normd)
                 ,facecolor='none',hatch = '..',edgecolor=edgecolors[1])
ax[2][1].fill_between(X,((data_MH['d']-data_MH['dd'])/normd)
                 ,((data_MH['d']+data_MH['dd'])/normd)
                 ,facecolor='none',hatch = '--',edgecolor=edgecolors[2])
ax[2][1].set_xlim(0.07,0.9)
ax[2][1].set_ylim([0.8,1.2])
ax[2][1].set_xlabel(r'\boldmath{$x$}',size=30)
ax[2][1].set_ylabel(r'\boldmath{${\rm ratio~to}~q_2$}',size=30,labelpad=10)
ax[2][1].tick_params(axis='both', which='both', labelsize=20,direction='in', length=6)

pp.savefig(fig,bbox_inches='tight')
pp.close()
