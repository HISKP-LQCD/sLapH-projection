#!/usr/bin/python

import math
import numpy as np
import scipy

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab

import boot
import fit
import IOcontraction
import plot

import sys

T = 48 # temporal extend of the lattice
L = 24

################################################################################
################ computing the fits ############################################

# array for timeslices
tt = []
for t in range(0, T):
	tt.append(float(t))
tt = np.asarray(tt)

# fitting the correlation function directly ####################################
sys.stdout = open('./fit_E0_4pt.out', 'w')
print '\n**********************************************************'
print ' Fitting the mass directly from the correlation function.'
print '**********************************************************'
plot_path = './plots/E0_4pt_fit_wsw.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot = PdfPages(plot_path)

# (2pt) reading data extracted from direct fit
############ p = 0 #############
filename = 'bootdata/C2_massfit_p0_median.npy'
C2_mass_median= np.load(filename)
p_squared = np.asarray([0, 1, 2, 3, 4, 5, 6, 8])*(2.*np.pi/L)**2

for p in range(0,5):
  fitresult, fitdata = [], []
  filename = 'bootdata/E0_corr_TP%d_GEVP_t1_wsw.npy' % p
  print '\n\n\n--------------------------------------'
  print 'PERFORMING FIT TO MOMENTUM:  %d' % p
  print '--------------------------------------'
  print 'reading file:', filename
  C2 = np.load(filename)
  C2_mean, C2_error = boot.mean_error_print(C2)
  print 'The correlation function:\n'
  i = 0
  for a, b in zip(C2_mean, C2_error):
    print i, a, '+/-', b
    i += 1

  deltaE = np.sqrt(np.power(C2_mass_median, 2) + p_squared[p])-C2_mass_median
  if p == 0:
    fitfunc1 = lambda p,t,tt: p[0]*(np.exp(-p[1]*(t+0.5)) - \
                                    np.exp(-p[1]*(T-t-0.5)))
    fitfunc2 = lambda p,t: p[0]*(np.exp(-p[1]*(t+0.5)) - \
                                    np.exp(-p[1]*(T-t-0.5)))
  else:
    m = deltaE[0]
    fitfunc1 = lambda p,t,tt: p[0]*(np.exp(-t*p[1])*(1.-np.exp(-p[1]+tt) + \
                                    np.exp(-(T-t)*p[1])*(1.-np.exp(p[1]+tt))))
    fitfunc2 = lambda p,t: p[0]*(np.exp(-t*p[1])*(1.-np.exp(-p[1]+m) + \
                                    np.exp(-(T-t)*p[1])*(1.-np.exp(p[1]+m))))


  counter = 0
  lo_lo =   [7,  7,  6,  5,  5]
  lo_up =   [20, 19, 14, 14, 13]
  up_lo =   [23, 22, 19, 16, 15]
  up_up =   [10, 10, 9,  8,  7]
  min_dof = [4,  4,  3,  3,  3]
  for lo in range(lo_lo[p], lo_up[p], 1):
    for up in range(up_lo[p], up_up[p], -1):
      if (up-lo-2) < min_dof[p]: # only fits with dof>1 are ok
        continue
      t = np.asarray(tt[lo:up+1])
      print "\nResult from bootstrapsample fit in interval (2pt):", lo, up
      start_parm = [C2_mean[5], np.log(C2_mean[5]/C2_mean[5+1])]
      res_C2, chi2, pvalue = fit.fitting(fitfunc1, t, \
                                    C2[:,lo:up+1], start_parm, 1, deltaE=deltaE)
      res_C2_mean = np.mean(res_C2, axis=0)
      label = ['t', 'E(t)', 'data', \
               'fit, tmin = %d, tmax = %d, p=%d, pval=%.4f' \
                % (lo, up, p, pvalue)]
      plot.corr_fct_with_fit(tt, C2_mean, C2_error, fitfunc2, res_C2_mean, \
                             [4., C2_mean.shape[0]], label, pdfplot, 1)
      # collecting fitresults
      if counter is 0:
        fitresult = res_C2[:,1]
        fitdata = np.array([chi2, pvalue, lo, up])
        counter += 1
      else:
        fitresult = np.vstack((fitresult, res_C2[:,1]))
        fitdata = np.vstack((fitdata, np.array([chi2, pvalue, lo, up])))
        counter += 1
#      sys.stdout.flush()
  # save fitted masses on disk
  path = 'bootdata/E0_4pt_massfit_p%d' % p
  IOcontraction.ensure_dir(path)
  np.save(path, fitresult)
  # save fitted masses on disk
  path = 'bootdata/E0_4pt_massfit_params_p%d' % p
  IOcontraction.ensure_dir(path)
  np.save(path, fitdata)

pdfplot.close()


# fitting the correlation function directly ####################################
sys.stdout = open('./fit_E0_4pt.out', 'w')
print '\n**********************************************************'
print ' Fitting the mass directly from the correlation function.'
print '**********************************************************'
plot_path = './plots/E0_4pt_fit_wsw.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot = PdfPages(plot_path)

# (2pt) reading data extracted from direct fit
############ p = 0 #############
filename = 'bootdata/C2_massfit_p0_median.npy'
C2_mass_median= np.load(filename)
p_squared = np.asarray([0, 1, 2, 3, 4, 5, 6, 8])*(2.*np.pi/L)**2

for p in range(0,5):
  fitresult, fitdata = [], []
  filename = 'bootdata/E0_corr_TP%d_GEVP_t1_wsw.npy' % p
  print '\n\n\n--------------------------------------'
  print 'PERFORMING FIT TO MOMENTUM:  %d' % p
  print '--------------------------------------'
  print 'reading file:', filename
  C2 = np.load(filename)
  C2_mean, C2_error = boot.mean_error_print(C2)
  print 'The correlation function:\n'
  i = 0
  for a, b in zip(C2_mean, C2_error):
    print i, a, '+/-', b
    i += 1

  deltaE = np.sqrt(np.power(C2_mass_median, 2) + p_squared[p])-C2_mass_median
  if p == 0:
    fitfunc1 = lambda p,t,tt: p[0]*(np.exp(-p[1]*(t+0.5)) - \
                                    np.exp(-p[1]*(T-t-0.5)))
    fitfunc2 = lambda p,t: p[0]*(np.exp(-p[1]*(t+0.5)) - \
                                    np.exp(-p[1]*(T-t-0.5)))
  else:
    m = deltaE[0]
    fitfunc1 = lambda p,t,tt: p[0]*(np.exp(-t*p[1])*(1.-np.exp(-p[1]+tt) + \
                                    np.exp(-(T-t)*p[1])*(1.-np.exp(p[1]+tt))))
    fitfunc2 = lambda p,t: p[0]*(np.exp(-t*p[1])*(1.-np.exp(-p[1]+m) + \
                                    np.exp(-(T-t)*p[1])*(1.-np.exp(p[1]+m))))


  counter = 0
  lo_lo =   [7,  7,  6,  5,  5]
  lo_up =   [14, 16, 14, 13, 10]
  up_lo =   [18, 20, 16, 14, 13]
  up_up =   [10, 10, 9,  8,  7]
  min_dof = [3,  3,  3,  3,  3]
  for lo in range(lo_lo[p], lo_up[p], 1):
    for up in range(up_lo[p], up_up[p], -1):
      if (up-lo-2) < min_dof[p]: # only fits with dof>1 are ok
        continue
      t = np.asarray(tt[lo:up+1])
      print "\nResult from bootstrapsample fit in interval (2pt):", lo, up
      start_parm = [C2_mean[5], np.log(C2_mean[5]/C2_mean[5+1])]
      res_C2, chi2, pvalue = fit.fitting(fitfunc1, t, \
                                    C2[:,lo:up+1], start_parm, 1, deltaE=deltaE)
      res_C2_mean = np.mean(res_C2, axis=0)
      label = ['t', 'E(t)', 'data', \
               'fit, tmin = %d, tmax = %d, p=%d, pval=%.4f' \
                % (lo, up, p, pvalue)]
      plot.corr_fct_with_fit(tt, C2_mean, C2_error, fitfunc2, res_C2_mean, \
                             [4., C2_mean.shape[0]], label, pdfplot, 1)
      # collecting fitresults
      if counter is 0:
        fitresult = res_C2[:,1]
        fitdata = np.array([chi2, pvalue, lo, up])
        counter += 1
      else:
        fitresult = np.vstack((fitresult, res_C2[:,1]))
        fitdata = np.vstack((fitdata, np.array([chi2, pvalue, lo, up])))
        counter += 1
#      sys.stdout.flush()
  # save fitted masses on disk
  path = 'bootdata/E0_4pt_massfit_p%d' % p
  IOcontraction.ensure_dir(path)
  np.save(path, fitresult)
  # save fitted masses on disk
  path = 'bootdata/E0_4pt_massfit_params_p%d' % p
  IOcontraction.ensure_dir(path)
  np.save(path, fitdata)

pdfplot.close()



