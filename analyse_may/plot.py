#!/usr/bin/python

import os, math, random, struct
import numpy as np
import scipy.optimize
import scipy.stats

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab

def corr_fct_with_fit(X, Y, dY, fitfunc, args, plotrange, label, pdfplot, logscale = 0):
  # plotting the data
  l = int(plotrange[0])
  u = int(plotrange[1])
  p1 = plt.errorbar(X[l:u], Y[l:u], dY[l:u], fmt='x' + 'b', label = label[4])
  # plotting the fit function
  x1 = np.linspace(l, u, 1000)
  y1 = []
  for i in x1:
    y1.append(fitfunc(args,i))
  y1 = np.asarray(y1)
  p2, = plt.plot(x1, y1, 'r', label = label[3])
  # adjusting the plot style
  plt.grid(True)
  plt.xlabel(label[0])
  plt.ylabel(label[1])
  plt.title(label[2])
  plt.legend([p1, p2], [label[3], label[4]], numpoints=1, fancybox=True, prop={'size':10})
  if logscale:
    plt.yscale('log')
  # save pdf
  pdfplot.savefig()
  plt.clf()


def hist(data, data_median, data_weight, data_std, data_sys_lo, data_sys_hi, \
         label, pdfplot):

  hist, bins = np.histogram(data[0:-1,0], 20, weights=data_weight[0:-1], density=True)
  width = 0.7 * (bins[1] - bins[0])
  center = (bins[:-1] + bins[1:]) / 2
  plt.xlabel(label[0])
  plt.ylabel(label[1])
  plt.title (label[2])

  plt.grid(True)
  x = np.linspace(center[0], center[-1], 1000)
  plt.plot(x, scipy.stats.norm.pdf(x, loc=data_median[0], scale=data_std), \
           'r-', lw=3, alpha=1, label='median + stat. error')
  plt.plot(x, scipy.stats.norm.pdf(x, loc=data_median[0], \
           scale=0.5*(data_sys_lo+data_sys_hi)),\
           'y-', lw=3, alpha=1, label='median + sys. error')
  plt.legend()
  plt.bar(center, hist, align='center', width=width, alpha=0.7)
  pdfplot.savefig()
  plt.clf()

# this can be used to plot the chisquare distribution of the fits
#  x = np.linspace(scipy.stats.chi2.ppf(1e-6, dof), scipy.stats.chi2.ppf(1.-1e-6, dof), 1000)
#  hist, bins = np.histogram(chisquare, 50, density=True)
#  width = 0.7 * (bins[1] - bins[0])
#  center = (bins[:-1] + bins[1:]) / 2
#  plt.xlabel('x')
#  plt.ylabel('chi^2(x)')
#  plt.grid(True)
#  plt.plot(x, scipy.stats.chi2.pdf(x, dof), 'r-', lw=2, alpha=1, label='chi2 pdf')
#  plt.bar(center, hist, align='center', width=width)
#  plt.show()

