#!/usr/bin/python

import math
import numpy as np
import scipy.optimize
import scipy.stats
from itertools import chain

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab

import IOcontraction

import sys
#sys.stdout = open('./dispersion_relation_pi.out', 'w')

# some global variables
################################################################################
momenta = range(0,5) # momenta to analyse
L = 24 # lattice extend
pvalue_cut = 0.0 # minimal p-value
min_fitrange = 2 # minimal fit range

# normalisation of a Matrix
################################################################################
def normalise(COV):
  norm = np.empty([COV.shape[0], COV.shape[1]], dtype=float)
  for i in range(0, COV.shape[0]):
    for j in range(0, COV.shape[1]):
      norm[i, j] = COV[i, j]/(np.sqrt(COV[i,i]*COV[j,j]))
  return norm

# choose correct p-values and fit ranges
# TODO: it is a little bit hacked to ensure the correct dimensions of M_cut
################################################################################
def choose_pvalue_cut(M, params, pcut, min_fitrange):
  counter = 0
  for i in range(0, params.shape[0]):
    if (params[i,1] >= pcut) and (params[i,1] <= 1.-pcut):
      if params[i,3] - params[i,2] > min_fitrange:
        if counter is 0:
          M_cut = M[i,:]
          params_out = params[i]
          counter += 1
        else:
          M_cut = np.vstack((M_cut, M[i,:]))
          params_out = np.vstack((params_out, params[i]))
  if (counter > 0):
    if (M_cut.shape[0] != M.shape[1]):
      return M_cut, params_out
    else:
      return np.vstack((M_cut, np.empty([0, M.shape[1]], dtype=float))), \
             params_out
  else:
    return np.empty([0, M.shape[1]], dtype=float), []


# compute weights
################################################################################
def compute_weight(corr, params):
  errors = np.std(corr, axis=1)
  max_err = np.amax(errors)
  if len(params) != 0:
    weights = []
    for i in range(0, params.shape[0]):
      w = (1.-abs(params[i,1]-0.5))*max_err/errors[i]
      weights.append(w**2)
    return weights
  else:
    return []

# compute the weighted quantile
################################################################################
def weighted_quantile(data, weights, quantile):
  ind_sorted = np.argsort(data)
  sorted_data = data[ind_sorted]
  sorted_weights = weights[ind_sorted]
  # Compute the auxiliary arrays
  Sn = np.cumsum(sorted_weights)
  Pn = (Sn-0.5*sorted_weights)/np.sum(sorted_weights)
  # Get the value of the weighted median
  interpolated_quant = np.interp(quantile, Pn, sorted_data)
  return interpolated_quant#, quant


################################################################################
################################################################################
################################################################################

##############################################################################
plot_path = './plots/Mpi_dispersion.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot = PdfPages(plot_path)

for p in momenta:

  # (2pt) reading data extracted from direct fit
  filename = 'bootdata/p%d/Mpi_p%d_fit.npy' % (p, p)
  C2_read = np.load(filename)
  filename = 'bootdata/p%d/Mpi_p%d_fit_params.npy' % (p, p)
  C2_params = np.load(filename)
 
  ##############################################################################
  # choosing combinations with wanted p-values and fit ranges
  C2_mass, C2_mass_params = \
           choose_pvalue_cut(C2_read, C2_params, pvalue_cut, min_fitrange)

  # compute weights
  C2_mass_weight = np.asarray(compute_weight(C2_mass, C2_mass_params))

  # median and std are needed later for the dispersion relation plot. Create 
  # array to save them
  if p == momenta[0]:
    C2_mass_median = np.empty((len(momenta), C2_mass.shape[1]), dtype=float)
    C2_mass_std = np.empty((len(momenta),1))
    C2_mass_sys_lo = np.empty((len(momenta),1))
    C2_mass_sys_hi = np.empty((len(momenta),1))

  # Creating the histogram
  for i in range(0, C2_mass.shape[1]):
    C2_mass_median[p,i] = weighted_quantile(C2_mass[0:-1,i]**2, \
                                                      C2_mass_weight[0:-1], 0.5)
  C2_mass_std[p] = np.std(C2_mass_median[p])
  C2_mass_16quant = weighted_quantile(C2_mass[0:-1,0]**2, \
                                                     C2_mass_weight[0:-1], 0.16)
  C2_mass_84quant = weighted_quantile(C2_mass[0:-1,0]**2, \
                                                     C2_mass_weight[0:-1], 0.84)
  
  C2_mass_sys_lo[p] = C2_mass_median[p,0] - C2_mass_16quant
  C2_mass_sys_hi[p] = C2_mass_84quant     - C2_mass_median[p,0]
  print 'pion mass for p=%d:' % p
  print '\tmedian +/- stat + sys - sys'
  print '\t(%.5e %.1e + %.1e - %.1e)\n' % (C2_mass_median[p,0], \
                           C2_mass_std[p], C2_mass_sys_hi[p], C2_mass_sys_lo[p])
  hist, bins = np.histogram(C2_mass[0:-1,0], 20, \
                                     weights=C2_mass_weight[0:-1], density=True)
  width = 0.7 * (bins[1] - bins[0])
  center = (bins[:-1] + bins[1:]) / 2
  # Plotting the histogram
  plt.xlabel(r'$m_{eff}$')
  plt.ylabel(r'weighted distribution of m_{eff}')
  plt.title('p=%d' % p)
  plt.grid(True)
  x = np.linspace(center[0], center[-1], 1000)
  plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median[p,0], 
           scale=C2_mass_std[p]), 'r-', lw=3, alpha=1, \
                                                   label='median + stat. error')
  plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median[p,0], \
           scale=0.5*(C2_mass_sys_lo[p]+C2_mass_sys_hi[p])), 'y-', lw=3, \
                                           alpha=1, label='median + sys. error')
  plt.legend()
  plt.bar(center, hist, align='center', width=width, alpha=0.7)
  pdfplot.savefig()
  plt.clf()

################################################################################
# Plotting the dispersion relation
l = -0.05
u = 0.30
plt.xlim([l, u])
plt.xlabel(r'$p^2$')
plt.ylabel(r'$E_{eff}^2$')
for p in momenta:
  C2_mass_std_sys_lo = np.sqrt(np.square(C2_mass_std[p]) + np.square(C2_mass_sys_lo[p]))
  C2_mass_std_sys_hi = np.sqrt(np.square(C2_mass_std[p]) + np.square(C2_mass_sys_hi[p]))
  plt.errorbar(p*(2.*np.pi/24)**2, C2_mass_median[p,0], \
                        yerr=[C2_mass_std_sys_lo, C2_mass_std_sys_hi],  fmt='b')
  plt.errorbar(p*(2.*np.pi/24)**2, C2_mass_median[p,0], C2_mass_std[p], \
                                                                  fmt='x' + 'b')
x1 = np.linspace(0, u, 1000)
y1 = []
for i in x1:
  y1.append(C2_mass_median[0,0] + i)
y1 = np.asarray(y1)
plt.plot(x1, y1, 'r')

# lattice dispersion relation Gattringer Lang
for i in range(4):
  plt.plot(i*np.square(2.*np.pi/24), np.square(np.arccosh(np.cosh(np.sqrt(C2_mass_median[0,0])) + \
                                              i*(1.-np.cos(2.*np.pi/24)))), \
                                              'gx', label = 'lattice dispersion relation')
plt.plot(4*np.square(2.*np.pi/24), np.square(np.arccosh(np.cosh(np.sqrt(C2_mass_median[0,0])) + \
                                             (1.-np.cos(2.*2.*np.pi/24)))), 'gx')
pdfplot.savefig()
plt.clf()

pdfplot.close()

## Plotting the dispersion relation
#l = -0.05
#u = 0.55
#plt.xlim([l, u])
#plt.xlabel(r'$p$')
#plt.ylabel(r'$E_{eff}(p)$')
#for p in momenta:
#  C2_mass_std_sys_lo = np.sqrt(np.square(C2_mass_std[p]) + \
#                                                   np.square(C2_mass_sys_lo[p]))
#  C2_mass_std_sys_hi = np.sqrt(np.square(C2_mass_std[p]) + \
#                                                   np.square(C2_mass_sys_hi[p]))
#  plt.errorbar(np.sqrt(p*(2.*np.pi/24)**2), C2_mass_median[p,0], \
#                        yerr=[C2_mass_std_sys_lo, C2_mass_std_sys_hi],  fmt='b')
#  plt.errorbar(np.sqrt(p*(2.*np.pi/24)**2), C2_mass_median[p,0], \
#                          C2_mass_std[p], label = 'lattice data', fmt='x' + 'b')
#x1 = np.linspace(0, u, 1000)
## continuum dispersion relation
#y1 = []
#for i in x1:
#  y1.append(np.sqrt(C2_mass_median[0,0]**2 + np.square(i)))
#y1 = np.asarray(y1)
#plt.plot(x1, y1, 'r')
#
## lattice dispersion relation Gattringer Lang
#for i in range(4):
#  plt.plot(np.sqrt(i)*2.*np.pi/24, np.arccosh(np.cosh(C2_mass_median[0,0]) + \
#                                              i*(1.-np.cos(2.*np.pi/24))), \
#                                              'gx', label = 'lattice dispersion relation')
#plt.plot(2*2.*np.pi/24, np.arccosh(np.cosh(C2_mass_median[0,0]) + \
#                                             (1.-np.cos(2.*2.*np.pi/24))), 'gx')
## approximation
#y2 = []
#for i in x1:
#  y2.append(np.arccosh(np.cosh(C2_mass_median[0,0]) + \
#                           np.square(24/(2.*np.pi)*i)*(1.-np.cos(2.*np.pi/24))))
#y2 = np.asarray(y2)
#plt.plot(x1, y2, 'g--')
#
#pdfplot.savefig()
#plt.clf()
#
#pdfplot.close()

## Plotting the dispersion relation E^2 vs. p^2, errors are broken (gaussian 
## error propagation)
#l = -0.05
#u = 0.55
#plt.xlim([l, u])
#plt.xlabel(r'$p^2$')
#plt.ylabel(r'$E_{eff}^2(p)$')
#for p in momenta:
#  C2_mass_std_sys_lo = np.square(C2_mass_std[p]) + np.square(C2_mass_sys_lo[p])
#  C2_mass_std_sys_hi = np.square(C2_mass_std[p]) + np.square(C2_mass_sys_hi[p])
#  plt.errorbar(p*(2.*np.pi/24)**2, np.square(C2_mass_median[p,0]), \
#                        yerr=[2*C2_mass_median[p,0]*C2_mass_std_sys_lo, 2*C2_mass_median[p,0]*C2_mass_std_sys_hi],  fmt='b')
#  plt.errorbar(p*(2.*np.pi/24)**2, np.square(C2_mass_median[p,0]), 2*C2_mass_median[p,0]*C2_mass_std[p], \
#                                                                  fmt='x' + 'b')
#x1 = np.linspace(0, u, 1000)
## continuum dispersion relation
#y1 = []
#for i in x1:
#  y1.append(C2_mass_median[0,0]**2 + i)
#y1 = np.asarray(y1)
#plt.plot(x1, y1, 'r')
#
## lattice dispersion relation Gattringer Lang
#for i in range(4):
#  plt.plot(i*np.square(2.*np.pi/24), np.square(np.arccosh(np.cosh(C2_mass_median[0,0]) + \
#                                              i*(1.-np.cos(2.*np.pi/24)))), 'gx')
#plt.plot(4*np.square(2.*np.pi/24), np.square(np.arccosh(np.cosh(C2_mass_median[0,0]) + \
#                                             (1.-np.cos(2.*2.*np.pi/24)))), 'gx')
## approximation
#y2 = []
#for i in x1:
#  y2.append(np.square(np.arccosh(np.cosh(C2_mass_median[0,0]) + \
#                           np.square(24/(2.*np.pi))*i*(1.-np.cos(2.*np.pi/24)))))
#y2 = np.asarray(y2)
#plt.plot(x1, y2, 'g--')
#
#pdfplot.savefig()
#plt.clf()
#
#pdfplot.close()

