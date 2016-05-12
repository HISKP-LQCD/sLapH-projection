#functions to calculate systematic error over fit intervals and plot histograms

import math
import numpy as np

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
  max_err = np.amin(errors)
  if len(params) != 0:
    weights = []
    for i in range(0, params.shape[0]):
      w = ((1.-2*abs(params[i,1]-0.5))*max_err/errors[i])**2
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

def compute_syst_err(data, params, write=1):
  ################################################################################
  # compute weights
  data_weight = np.asarray(compute_weight(data, params))
  
  ################################################################################
  data_median = np.empty([data.shape[1]], dtype=float)
  for i in range(0, data.shape[1]):
    data_median[i] = weighted_quantile(data[0:-1,i], data_weight[0:-1], 0.5)
  data_std = np.std(data_median)
  data_16quant = weighted_quantile(data[0:-1,0], data_weight[0:-1], 0.16)
  data_84quant = weighted_quantile(data[0:-1,0], data_weight[0:-1], 0.84)
  
  data_sys_lo = data_median[0] - data_16quant
  data_sys_hi = data_84quant   - data_median[0]
  
  if write:
    print '\tmedian +/- stat + sys - sys'
    print '\t(%.5e %.1e + %.1e - %.1e)\n' % (data_median[0], data_std, \
                                             data_sys_hi, data_sys_lo)
  return data_median, data_weight, data_std, data_sys_lo, data_sys_hi


