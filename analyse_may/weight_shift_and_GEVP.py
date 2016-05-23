#!/usr/bin/python

import numpy as np

import boot
import IOcontraction
import gevp

import sys
#sys.stdout = open('./weight_shift_and_GEVP.out', 'w')

# some global variables
################################################################################
momenta = range(0,1) # momenta to analyse
t0s = range(1,5) # t0's for gevp

L = 24 # lattice extend
T = 48
pvalue_cut = 0.0 # minimal p-value
min_fitrange = 2 # minimal fit range

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

    E = []
    E_sw = []
    for t0 in t0s:
      # Gevp on original correlators
      E.append(gevp.calculate_gevp(Crho, t0))
  
      # Gevp on shifted correlators
      E_sw.append(gevp.calculate_gevp(Crho_shifted, t0))

    E = np.asarray(E)
    E_sw = np.asarray(E_sw)

    print E.shape
    print E_sw.shape

    if E.shape[-1] != E_sw.shape[-1]:
      print 'Gevp yielded different number of eigenvalues with and without ' \
            'shift+weight.'
      exit(0)
    nb_eigenvalues = E.shape[-1]

    # write data to disc
    IOcontraction.ensure_dir('./bootdata')
    IOcontraction.ensure_dir('./bootdata/p%d' % p)
    IOcontraction.ensure_dir('./bootdata/p%d/without_shift_weight' % p)

    filename = './bootdata/p%d/without_shift_weight/Crho_p%d_sym_gevp' % (p, p)
    np.save(filename, E)

    filename = './bootdata/p%d/Crho_p%d_sym_sw_gevp' % (p, p)
    np.save(filename, E_sw)
     
