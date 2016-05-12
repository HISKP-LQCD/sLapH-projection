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
sys.stdout = open('./dispersion_relation_pi.out', 'w')

# some global variables
################################################################################
p = 0 # momenta to analyse
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
# (2pt) reading data extracted from direct fit
############ p = 0 #############
filename = 'bootdata/C2_massfit_p0.npy'
C2_read0 = np.load(filename)
filename = 'bootdata/C2_massfit_params_p0.npy'
C2_params0 = np.load(filename)
############ p = 1 #############
filename = 'bootdata/C2_massfit_p1.npy'
C2_read1 = np.load(filename)
filename = 'bootdata/C2_massfit_params_p1.npy'
C2_params1 = np.load(filename)
############ p = 2 #############
filename = 'bootdata/C2_massfit_p2.npy'
C2_read2 = np.load(filename)
filename = 'bootdata/C2_massfit_params_p2.npy'
C2_params2 = np.load(filename)
############ p = 3 #############
filename = 'bootdata/C2_massfit_p3.npy'
C2_read3 = np.load(filename)
filename = 'bootdata/C2_massfit_params_p3.npy'
C2_params3 = np.load(filename)

# choosing combinations with wanted p-values and fit ranges
################################################################################
C2_mass0, C2_mass_params0 = \
         choose_pvalue_cut(C2_read0, C2_params0, pvalue_cut, min_fitrange)
C2_mass1, C2_mass_params1 = \
         choose_pvalue_cut(C2_read1, C2_params1, pvalue_cut, min_fitrange)
C2_mass2, C2_mass_params2 = \
         choose_pvalue_cut(C2_read2, C2_params2, pvalue_cut, min_fitrange)
C2_mass3, C2_mass_params3 = \
         choose_pvalue_cut(C2_read3, C2_params3, pvalue_cut, min_fitrange)

# compute weights
################################################################################
C2_mass_weight0 = np.asarray(compute_weight(C2_mass0, C2_mass_params0))
C2_mass_weight1 = np.asarray(compute_weight(C2_mass1, C2_mass_params1))
C2_mass_weight2 = np.asarray(compute_weight(C2_mass2, C2_mass_params2))
C2_mass_weight3 = np.asarray(compute_weight(C2_mass3, C2_mass_params3))

# writing the 2pt pion mass out and creating a histogram with all data
################################################################################
plot_path = './plots/Pion_dispersion.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot = PdfPages(plot_path)
################################################################################
############ p = 0 #############
C2_mass_median0 = np.empty([C2_mass0.shape[1]], dtype=float)
for i in range(0, C2_mass0.shape[1]):
  C2_mass_median0[i] = weighted_quantile(C2_mass0[0:-1,i], C2_mass_weight0[0:-1], 0.5)
C2_mass_std0 = np.std(C2_mass_median0)
C2_mass_16quant0 = weighted_quantile(C2_mass0[0:-1,0], C2_mass_weight0[0:-1], 0.16)
C2_mass_84quant0 = weighted_quantile(C2_mass0[0:-1,0], C2_mass_weight0[0:-1], 0.84)

C2_mass_sys_lo0 = C2_mass_median0[0] - C2_mass_16quant0
C2_mass_sys_hi0 = C2_mass_84quant0   - C2_mass_median0[0]
print 'pion mass for p=0:'
print '\tmedian +/- stat + sys - sys'
print '\t(%.5e %.1e + %.1e - %.1e)\n' % (C2_mass_median0[0], C2_mass_std0, C2_mass_sys_hi0, C2_mass_sys_lo0)
hist, bins = np.histogram(C2_mass0[0:-1,0], 20, weights=C2_mass_weight0[0:-1], density=True)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.xlabel(r'$m_{eff}$')
plt.ylabel(r'weighted distribution of m_{eff}')
plt.title('p=0')
plt.grid(True)
x = np.linspace(center[0], center[-1], 1000)
plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median0[0], scale=C2_mass_std0),\
         'r-', lw=3, alpha=1, label='median + stat. error')
plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median0[0], \
         scale=0.5*(C2_mass_sys_lo0+C2_mass_sys_hi0)),\
         'y-', lw=3, alpha=1, label='median + sys. error')
plt.legend()
plt.bar(center, hist, align='center', width=width, alpha=0.7)
pdfplot.savefig()
plt.clf()
############ p = 1 #############
C2_mass_median1 = np.empty([C2_mass1.shape[1]], dtype=float)
for i in range(0, C2_mass1.shape[1]):
  C2_mass_median1[i] = weighted_quantile(C2_mass1[0:-1,i], C2_mass_weight1[0:-1], 0.5)
C2_mass_std1 = np.std(C2_mass_median1)
C2_mass_16quant1 = weighted_quantile(C2_mass1[0:-1,0], C2_mass_weight1[0:-1], 0.16)
C2_mass_84quant1 = weighted_quantile(C2_mass1[0:-1,0], C2_mass_weight1[0:-1], 0.84)

C2_mass_sys_lo1 = C2_mass_median1[0] - C2_mass_16quant1
C2_mass_sys_hi1 = C2_mass_84quant1   - C2_mass_median1[0]
print 'pion mass for p=1:'
print '\tmedian +/- stat + sys - sys'
print '\t(%.5e %.1e + %.1e - %.1e)\n' % (C2_mass_median1[0], C2_mass_std1, C2_mass_sys_hi1, C2_mass_sys_lo1)
hist, bins = np.histogram(C2_mass1[0:-1,0], 20, weights=C2_mass_weight1[0:-1], density=True)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.xlabel(r'$m_{eff}$')
plt.ylabel(r'weighted distribution of m_{eff}')
plt.title('p=1')
plt.grid(True)
x = np.linspace(center[0], center[-1], 1000)
plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median1[0], scale=C2_mass_std1),\
         'r-', lw=3, alpha=1, label='median + stat. error')
plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median1[0], \
         scale=0.5*(C2_mass_sys_lo1+C2_mass_sys_hi1)),\
         'y-', lw=3, alpha=1, label='median + sys. error')
plt.legend()
plt.bar(center, hist, align='center', width=width, alpha=0.7)
pdfplot.savefig()
plt.clf()
############ p = 2 #############
C2_mass_median2 = np.empty([C2_mass2.shape[1]], dtype=float)
for i in range(0, C2_mass2.shape[1]):
  C2_mass_median2[i] = weighted_quantile(C2_mass2[0:-1,i], C2_mass_weight2[0:-1], 0.5)
C2_mass_std2 = np.std(C2_mass_median2)
C2_mass_16quant2 = weighted_quantile(C2_mass2[0:-1,0], C2_mass_weight2[0:-1], 0.16)
C2_mass_84quant2 = weighted_quantile(C2_mass2[0:-1,0], C2_mass_weight2[0:-1], 0.84)

C2_mass_sys_lo2 = C2_mass_median2[0] - C2_mass_16quant2
C2_mass_sys_hi2 = C2_mass_84quant2   - C2_mass_median2[0]
print 'pion mass for p=2:'
print '\tmedian +/- stat + sys - sys'
print '\t(%.5e %.1e + %.1e - %.1e)\n' % (C2_mass_median2[0], C2_mass_std2, C2_mass_sys_hi2, C2_mass_sys_lo2)
hist, bins = np.histogram(C2_mass2[0:-1,0], 20, weights=C2_mass_weight2[0:-1], density=True)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.xlabel(r'$m_{eff}$')
plt.ylabel(r'weighted distribution of m_{eff}')
plt.title('p=2')
plt.grid(True)
x = np.linspace(center[0], center[-1], 1000)
plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median2[0], scale=C2_mass_std2),\
         'r-', lw=3, alpha=1, label='median + stat. error')
plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median2[0], \
         scale=0.5*(C2_mass_sys_lo2+C2_mass_sys_hi2)),\
         'y-', lw=3, alpha=1, label='median + sys. error')
plt.legend()
plt.bar(center, hist, align='center', width=width, alpha=0.7)
pdfplot.savefig()
plt.clf()
############ p = 3 #############
C2_mass_median3 = np.empty([C2_mass3.shape[1]], dtype=float)
for i in range(0, C2_mass3.shape[1]):
  C2_mass_median3[i] = weighted_quantile(C2_mass3[0:-1,i], C2_mass_weight3[0:-1], 0.5)
C2_mass_std3 = np.std(C2_mass_median3)
C2_mass_16quant3 = weighted_quantile(C2_mass3[0:-1,0], C2_mass_weight3[0:-1], 0.16)
C2_mass_84quant3 = weighted_quantile(C2_mass3[0:-1,0], C2_mass_weight3[0:-1], 0.84)

C2_mass_sys_lo3 = C2_mass_median3[0] - C2_mass_16quant3
C2_mass_sys_hi3 = C2_mass_84quant3   - C2_mass_median3[0]
print 'pion mass for p=3:'
print '\tmedian +/- stat + sys - sys'
print '\t(%.5e %.1e + %.1e - %.1e)\n' % (C2_mass_median3[0], C2_mass_std3, C2_mass_sys_hi3, C2_mass_sys_lo3)
hist, bins = np.histogram(C2_mass3[0:-1,0], 20, weights=C2_mass_weight3[0:-1], density=True)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.xlabel(r'$m_{eff}$')
plt.ylabel(r'weighted distribution of m_{eff}')
plt.title('p=3')
plt.grid(True)
x = np.linspace(center[0], center[-1], 1000)
plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median3[0], scale=C2_mass_std3),\
         'r-', lw=3, alpha=1, label='median + stat. error')
plt.plot(x, scipy.stats.norm.pdf(x, loc=C2_mass_median3[0], \
         scale=0.5*(C2_mass_sys_lo3+C2_mass_sys_hi3)),\
         'y-', lw=3, alpha=1, label='median + sys. error')
plt.legend()
plt.bar(center, hist, align='center', width=width, alpha=0.7)
pdfplot.savefig()
plt.clf()


# Plotting the dispersion relation
l = -0.05
u = 0.25
plt.xlim([l, u])
plt.xlabel(r'$p^2$')
plt.ylabel(r'$E_{eff}$')
plt.errorbar(0,                  C2_mass_median0[0], C2_mass_std0, fmt='x' + 'b')
plt.errorbar((2.*np.pi/24)**2,   C2_mass_median1[0], C2_mass_std1, fmt='x' + 'b')
plt.errorbar(2*(2.*np.pi/24)**2, C2_mass_median2[0], C2_mass_std2, fmt='x' + 'b')
plt.errorbar(3*(2.*np.pi/24)**2, C2_mass_median3[0], C2_mass_std3, fmt='x' + 'b')

x1 = np.linspace(0, u, 1000)
y1 = []
for i in x1:
  y1.append(np.sqrt(C2_mass_median0[0]**2 + i))
y1 = np.asarray(y1)
plt.plot(x1, y1, 'r')
pdfplot.savefig()
plt.clf()


pdfplot.close()


