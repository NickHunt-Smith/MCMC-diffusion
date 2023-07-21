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

outdir = 'Gaussian_2D'
dim = 2
low_bound = np.zeros(dim)
high_bound = 10*np.ones(dim)

samples_total = 51000
samples_burn_in = 1000

algo1_samples = np.loadtxt(outdir + '/samples.dat')
comparison_samples = np.loadtxt(outdir + '/comparison_samples.dat')

algo1_samples = np.array(algo1_samples[samples_burn_in:samples_total])
comparison_samples = np.array(comparison_samples[samples_burn_in:samples_total])

algo1_samples = np.vstack((algo1_samples,[low_bound,high_bound]))
comparison_samples = np.vstack((comparison_samples,[low_bound,high_bound]))

Ranges = ((0,10),(0,10))
names = [r'\boldmath{$\theta_1$}',r'\boldmath{$\theta_2$}']
chainLabels = [r'\rm \bf Algorithm 1', r'\rm \bf Pure MH']
GTC = pygtc.plotGTC(chains=[algo1_samples,comparison_samples],nContourLevels = 3,paramRanges = Ranges,do1dPlots=True,nBins  = 50,paramNames = names, chainLabels = chainLabels,figureSize = 5, customLabelFont = {'family':'Times New Roman', 'size':20}, customLegendFont = {'family':'Times New Roman', 'size':20}, customTickFont = {'family':'Times New Roman', 'size':10},legendMarker='None')
GTC.savefig(outdir + '/comparison_plot.pdf',bbox_inches='tight')
