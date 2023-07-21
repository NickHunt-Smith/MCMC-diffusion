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

outdir = 'Rosenbrock_4D'
dim = 4
low_bound = -3*np.ones(dim)
high_bound = 3*np.ones(dim)

samples_total = 200000
samples_burn_in = 10000

algo1_samples = np.loadtxt(outdir + '/samples.dat')
comparison_samples = np.loadtxt(outdir + '/comparison_samples.dat')

algo1_samples = np.array(algo1_samples[samples_burn_in:samples_total])
comparison_samples = np.array(comparison_samples[samples_burn_in:samples_total])

algo1_samples = np.vstack((algo1_samples,[low_bound,high_bound]))
comparison_samples = np.vstack((comparison_samples,[low_bound,high_bound]))

Ranges = ((-1,1.5),(-0.5,1.5),(-0.5,1.75),(-0.5,3))
names = [r'\boldmath{$\theta_1$}',r'\boldmath{$\theta_2$}',r'\boldmath{$\theta_3$}',r'\boldmath{$\theta_4$}']
chainLabels = [r'\rm \bf Algorithm 1', r'\rm \bf Pure MH']
GTC = pygtc.plotGTC(chains=[algo1_samples,comparison_samples],nContourLevels = 3,paramRanges = Ranges,do1dPlots=True,nBins  = 50,paramNames = names, chainLabels = chainLabels,figureSize = 8, customLabelFont = {'family':'Times New Roman', 'size':25}, customLegendFont = {'family':'Times New Roman', 'size':25}, customTickFont = {'family':'Times New Roman', 'size':15})
GTC.savefig(outdir + '/comparison_plot.pdf',bbox_inches='tight')
