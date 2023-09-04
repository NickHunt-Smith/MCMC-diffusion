import sys,os
import math
import numpy as np
import time
import pygtc
import scipy

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

outdir = 'Gaussian_10D'
dim = 10
low_bound = -5*np.ones(dim)
high_bound = 10*np.ones(dim)

samples_total = 10000
samples_burn_in = 100

algo1_samples = np.loadtxt(outdir + '/samples.dat')

algo1_samples = np.array(algo1_samples[samples_burn_in:samples_total])

algo1_samples = np.vstack((algo1_samples,[low_bound,high_bound]))

Ranges = ((-5,10),(-5,10),(-5,10),(-5,10),(-5,10),(-5,10),(-5,10),(-5,10),(-5,10),(-5,10))
names = [r'\boldmath{$\theta_1$}',r'\boldmath{$\theta_2$}',r'\boldmath{$\theta_3$}',r'\boldmath{$\theta_4$}',r'\boldmath{$\theta_5$}',r'\boldmath{$\theta_6$}',r'\boldmath{$\theta_7$}',r'\boldmath{$\theta_8$}',r'\boldmath{$\theta_9$}',r'\boldmath{$\theta_{10}$}']
chainLabels = [r'\rm \bf Algorithm 1']
truth = ((-10,-10,-10,-10,-10,-10,-10,-10,-10,-10))
GTC = pygtc.plotGTC(chains=[algo1_samples],nContourLevels = 3,paramRanges = Ranges,do1dPlots=True,nBins  = 50,paramNames = names, chainLabels = chainLabels,figureSize = 5, customLabelFont = {'family':'Times New Roman', 'size':15}, customLegendFont = {'family':'Times New Roman', 'size':20}, customTickFont = {'family':'Times New Roman', 'size':5},legendMarker='None',truths = truth,truthLabels=(r'\rm \bf True'),truthColors = ('r'),truthLineStyles = '-')

GTC.savefig(outdir + '/comparison_plot.pdf',bbox_inches='tight')
