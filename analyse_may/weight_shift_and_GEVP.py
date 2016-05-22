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

import boot
import IOcontraction
import gevp

import sys
#sys.stdout = open('./weight_shift_and_GEVP.out', 'w')

# some global variables
################################################################################
momenta = range(0,1) # momenta to analyse
t0s = range(1,6) # t0's for gevp

L = 24 # lattice extend
T = 48
pvalue_cut = 0.0 # minimal p-value
min_fitrange = 2 # minimal fit range

# for plotting multiple plots on one page
################################################################################
def example_plot(ax, fontsize, xlabel, ylabel, title):
  ax.set_yscale('log')
  ax.grid(True)
  ax.set_xlabel(xlabel, fontsize=fontsize)
  ax.set_ylabel(ylabel, fontsize=fontsize)
  ax.set_title(title, fontsize=fontsize)
  ax.legend(numpoints=1, fancybox=True, prop={'size':fontsize}, loc=3)

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
#filename = 'bootdata/C2_massfit_p0.npy'
#C2_read = np.load(filename)
#filename = 'bootdata/C2_massfit_params_p0.npy'
#C2_params = np.load(filename)
## choosing combinations with wanted p-values and fit ranges
#C2_mass, C2_mass_params = \
#         choose_pvalue_cut(C2_read, C2_params, pvalue_cut, min_fitrange)
## compute weights
#C2_mass_weight = np.asarray(compute_weight(C2_mass, C2_mass_params))
## compute medians
#C2_mass_median = np.empty([C2_mass.shape[1]], dtype=float)
#for i in range(0, C2_mass.shape[1]):
#  C2_mass_median[i] = weighted_quantile(C2_mass[0:-1,i], \
#                                        C2_mass_weight[0:-1], 0.5)
#path = 'bootdata/C2_massfit_p0_median.npy'
#IOcontraction.ensure_dir(path)
#np.save(path, C2_mass_median)

p_squared = np.asarray([0, 1, 2, 3, 4, 5, 6, 8])*(2.*np.pi/L)**2

for p in momenta:

  # reading data
  filename = 'bootdata/p%d/Crho_p%d_sym.npy' % (p, p)
  data = np.load(filename)

  for Crho in data:

    # weighting
    Crho_weighted = np.copy(Crho)
  
  #  if p is not 0:
  #    for b in range(0, rho_00_w.shape[0]):
  #      for t in range(0, rho_00_w.shape[1]):
  #        deltaE = np.sqrt(C2_mass_median[b]**2 + p_squared[p])-C2_mass_median[b]
  #        rho_00_w[b, t] = rho_00_w[b, t] * np.exp(t*deltaE) 
  #        rho_01_w[b, t] = rho_01_w[b, t] * np.exp(t*deltaE) 
  #        rho_11_w[b, t] = rho_11_w[b, t] * np.exp(t*deltaE) 
  
    # shifting
    Crho_shifted, a, b = boot.compute_derivative(Crho_weighted)
  
  #  # re-weighting
  #  if p is not 0:
  #    for b in range(0, rho_00_shift.shape[0]):
  #      for t in range(0, rho_00_shift.shape[1]):
  #        deltaE = np.sqrt(C2_mass_median[b]**2 + p_squared[p])-C2_mass_median[b]
  #        rho_00_shift[b, t] = rho_00_shift[b, t] * np.exp(-t*deltaE) 
  #        rho_01_shift[b, t] = rho_01_shift[b, t] * np.exp(-t*deltaE) 
  #        rho_11_shift[b, t] = rho_11_shift[b, t] * np.exp(-t*deltaE) 
  
    #GEVP
    # Plotting
    plot_path = './plots/Energy_eigenvalues_p%d.pdf' % p
    IOcontraction.ensure_dir(plot_path)
    pdfplot = PdfPages(plot_path)
  
    for t0 in t0s:
      E = gevp.calculate_gevp(Crho, t0)
      E_mean, E_std = boot.mean_error_print(E)
  
      E_sw = gevp.calculate_gevp(Crho_shifted, t0)
      E_mean_sw, E_std_sw = boot.mean_error_print(E_sw)
  #    E0_sw, E1_sw = gevp.calculate_gevp(rho_00_shift, rho_01_shift, \
  #                                       rho_11_shift, t0)
  #    E0_mean_sw, E0_std_sw = boot.mean_error_print(E0_sw)
  #    E1_mean_sw, E1_std_sw = boot.mean_error_print(E1_sw)
  
  #    E0, E1 = gevp.calculate_gevp(rho_00, rho_01, rho_11, t0)
  #    E0_mean, E0_std = boot.mean_error_print(E0)
  #    E1_mean, E1_std = boot.mean_error_print(E1)
      if E.shape[-1] != E_sw.shape[-1]:
        print 'Gevp yielded different number of eigenvalues with and without ' \
              'shift+weight.'
        exit(0)
      nb_eigenvalues = E.shape[-1]
      print nb_eigenvalues

        xlabel = r'$t/a$'
        ylabel = r'Energie level - $aE_{\pi\pi}(t/a)$'
        title = r'GEVP $t_0 = %d$, $P = %d$' % (t0, p)
        f, (ax1, ax3) = plt.subplots(2)
        plt.suptitle(title, fontsize=13)
 
      for n in range(nb_eigenvalues):
        if t0 == 1:
          tmp = ['First', 'Second', 'Third', 'Fourth', 'Fifth']
          print '\n%s eigenvalues for t_0 = 1 and p = %d\n' % (tmp[n], p)
          boot.mean_error_print(E_sw[...,n], 1) 
          print '\n\n'
    
        #######
        ax1.errorbar(range(0, len(E_mean)), E_mean[...,n], E_std[...,n], \
                     fmt='x', label=r'$E_%d$' % n)
        example_plot(ax1, 11, xlabel, ylabel, r'Gevp without shift and weight')
        #######
  #      ax2.errorbar(range(0, len(E0_mean_sw)), E0_mean_sw, E0_std_sw, \
  #                   fmt='x' + 'c', label=r'$E_0$')
  #      ax2.errorbar(range(0, len(E1_mean_sw)), E1_mean_sw, E1_std_sw, \
  #                   fmt='x' + 'k', label=r'$E_1$')
  #      example_plot(ax2, 11, xlabel, ylabel, r'GEVP with shift and weight')
        #######
        ax3.errorbar(range(0, len(E_mean_sw)), E_mean_sw[...,n], E_std_sw[...,n], \
                     fmt='x', label=r'$E_%d$' % n)
        example_plot(ax3, 11, xlabel, ylabel, r'Gevp with shift and weight')
  
      plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
      plt.subplots_adjust(top=0.90)
      pdfplot.savefig()
      plt.clf()
  
  #    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
  #    ax3.errorbar(range(0, len(E0_mean_sw)), E0_mean_sw/E0_mean_sw[10], \
  #                 E0_std_sw/E0_mean_sw[10], fmt='x' + 'c', label=r'w. s+w')
  #    ax3.errorbar(range(0, len(E0_mean)), E0_mean/E0_mean[10], \
  #                 E0_std/E0_mean[10], fmt='x' + 'b', label=r'w.o. s+w')
  #    example_plot(ax3, 11, xlabel, ylabel, r'GEVP normalised first ev $(E_0)$')
  #    #######
  #    ax4.errorbar(range(0, len(E1_mean_sw)), E1_mean_sw/E1_mean_sw[10], \
  #                 E1_std_sw/E1_mean_sw[10], fmt='x' + 'k', label=r'w. s+w')
  #    ax4.errorbar(range(0, len(E1_mean)), E1_mean/E1_mean[10], \
  #                 E1_std/E1_mean[10], fmt='x' + 'r', label=r'w.o. s+w')
  #    example_plot(ax4, 11, xlabel, ylabel, r'GEVP normalised second ev $(E_1)$')
  #    #######
    
  #    # write data to disc
  #    path = './bootdata/E0_corr_TP%d_GEVP_t%d_wsw' % (p, t0)
  #    IOcontraction.ensure_dir(path)
  #    np.save(path, E0_sw)
  #    path = './bootdata/E1_corr_TP%d_GEVP_t%d_wsw' % (p, t0)
  #    IOcontraction.ensure_dir(path)
  #    np.save(path, E1_sw)
  #    path = './bootdata/E0_corr_TP%d_GEVP_t%d_wosw' % (p, t0)
  #    IOcontraction.ensure_dir(path)
  #    np.save(path, E0)
  #    path = './bootdata/E1_corr_TP%d_GEVP_t%d_wosw' % (p, t0)
  #    IOcontraction.ensure_dir(path)
  #    np.save(path, E1)
    
    pdfplot.close()



