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
#sys.stdout = open('./fit_pion_2pt.out', 'w')

T = 48 # temporal extend of the lattice

################################################################################
################ computing the fits ############################################

# array for timeslices
tt = []
for t in range(0, T):
  tt.append(float(t))
tt = np.asarray(tt)

# fitting the correlation function directly ####################################
print '\n**********************************************************'
print ' Fitting the mass directly from the correlation function.'
print '**********************************************************'

plot_path = './plots/Mpi_fit.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot = PdfPages(plot_path)

#for p in range(0,5):
for p in range(0,1):
  filename = './bootdata/p%d/Mpi_p%d_sym.npy' % (p, p)
  print 'reading file:', filename
  C2 = np.load(filename)
  print C2.shape
  C2_mean, C2_error = boot.mean_error_print(C2)
  m, m_mean, m_error = boot.compute_mass(C2) 
  print 'The correlation function:\n'
  i = 0
  for a, b in zip(C2_mean, C2_error):
    print i, a, '+/-', b
    i += 1

  print '\n'

  print 'The effective mass:\n'
  i = 0
  for a, b in zip(m_mean, m_error):
    print i, a, '+/-', b
    i += 1

#  min_t_of_p = [14, 12, 10, 8]
#  max_t_of_p = [C2.shape[1], C2.shape[1], 24, 19]
  counter = 0
  lo_lo =   [11,  7,  6,  5,  5]
  lo_up =   [18, 16, 14, 13, 10]
  up_lo =   [23, 20, 16, 14, 13]
  up_up =   [15, 10, 9,  8,  7]
  min_dof = [5,  3,  3,  3,  3]

  fitfunc1 = lambda p,t,tt: p[0]*(np.exp(-t*p[1]) + np.exp(-(T-t)*p[1]))
  fitfunc2 = lambda p,t: p[0]

#  for lo in range(min_t_of_p[p], max_t_of_p[p], 2):
#    for up in range(max_t_of_p[p], min_t_of_p[p], -2):
#      if (up-lo-2) < 4: # only fits with dof>1 are ok
#        continue
  for lo in range(lo_lo[p], lo_up[p], 1):
    for up in range(up_lo[p], up_up[p], -1):
      if (up-lo-2) < min_dof[p]: # only fits with dof>1 are ok
        continue
      t = np.asarray(tt[lo:up+1])
      print "\nResult from bootstrapsample fit in interval (2pt):", lo, up
      start_parm = [C2_mean[lo], m_mean[lo]]
      res_C2, chi2, dof, pvalue = fit.fitting(fitfunc1, t, C2[:,lo:up+1], \
                                                                     start_parm)
      res_C2_mean = np.mean(res_C2, axis=0)
      label = ['t', r'$m_{\pi}^{eff}(t)$', 'Pion mass fits for $p_{cm} = %d$,' \
               '$ t \in [%d, %d]$' % (p, lo, up), 'data', \
               '$m_{\pi}^{eff} = %f$\n$\chi^2/d.o.f. = %f$\n' \
                                'p-value: $%f$' % (res_C2_mean[1], chi2/dof, pvalue)]
      plot.corr_fct_with_fit(tt, m_mean, m_error, fitfunc2, [res_C2_mean[1]], \
                             [7., m.shape[1]], label, pdfplot, 0)
      # collecting fitresults
      if counter is 0:
        fitresult = res_C2[:,1]
        fitdata = np.array([chi2, pvalue, lo, up])
        counter += 1
      else:
        fitresult = np.vstack((fitresult, res_C2[:,1]))
        fitdata = np.vstack((fitdata, np.array([chi2, pvalue, lo, up-1])))
        counter += 1
      sys.stdout.flush()

  # save fitted masses on disk
  path = 'bootdata/p%d/Mpi_p%d_fit' % (p, p)
  IOcontraction.ensure_dir(path)
  np.save(path, fitresult)
  # save fitted masses on disk
  path = 'bootdata/p%d/Mpi_p%d_fit_params' % (p, p)
  IOcontraction.ensure_dir(path)
  np.save(path, fitdata)

pdfplot.close()





