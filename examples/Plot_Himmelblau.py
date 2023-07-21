import sys,os
import math
import numpy as np
import time
import pygtc

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
matplotlib.rc('text', usetex = True)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import pylab as py
from matplotlib.backends.backend_pdf import PdfPages

outdir = 'Himmelblau_2D'
dim = 2

samples_total = 12000
samples_burn_in = 2000

algo1_samples = np.loadtxt(outdir + '/samples.dat')
algo1_samples = np.array(algo1_samples[samples_burn_in:samples_total])
algo1_samples = np.vstack((algo1_samples,[[-5,-5],[-5,5],[5,-5],[5,5]]))
Ranges = ((-5,5),(-5,5))
GTC = pygtc.plotGTC(chains=algo1_samples,nContourLevels = 3,paramRanges = Ranges,do1dPlots=False,nBins  = 50,figureSize = 10)

# Remove everything below until GTC.savefig if you don't want to plot the jumps between modes
ax = GTC.axes
algo1_samples_diff_only = np.loadtxt(outdir + '/samples.dat')
diff = [algo1_samples_diff_only[0]]
jumps = 0
for i in range(len(algo1_samples_diff_only[:,0])):
    difference = ((algo1_samples_diff_only[i,0] - algo1_samples_diff_only[i-1,0])**2) + ((algo1_samples_diff_only[i,1] - algo1_samples_diff_only[i-1,1])**2)
    if difference > 4:
        diff.append(algo1_samples_diff_only[i])
        jumps += 1
print('Number of mode jumps from diffusion proposal = ',str(jumps))

diff = np.array(diff)
ax[0].plot(diff[:,0],diff[:,1],lw = 1,color = 'k',label = r'\rm \bf mode jumps')
ax[0].scatter(diff[:,0],diff[:,1],s = 15,color = 'k')
ax[0].fill_between([-7,-6],[-7,-6],color = '#1f77b4',label = r'\boldmath{$1 \sigma$} \rm \bf contour')
ax[0].fill_between([-8,-7],[-7,-6],color = '#52aae7',label = r'\boldmath{$2 \sigma$} \rm \bf contour')
ax[0].fill_between([-9,-8],[-7,-6],color = '#85ddff',label = r'\boldmath{$3 \sigma$} \rm \bf contour')
ax[0].set_xlabel(r'\boldmath{$\theta_1$}',size = 30)
ax[0].set_ylabel(r'\boldmath{$\theta_2$}',size = 30)
ax[0].set_ylim(-5.5,5)
ax[0].set_xlim(-5.5,5.5)
ax[0].tick_params(axis='both', which='major', labelsize=25,direction='in', length=6)
ax[0].legend(frameon = 0,fontsize = 20,loc = 'lower right')

GTC.savefig(outdir + '/jumps_plot.pdf',bbox_inches='tight')
