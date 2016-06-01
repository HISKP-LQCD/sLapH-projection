#!/usr/bin/python

import math, random
import numpy as np

import IOcontraction

# bootstrapping
def bootstrap(X, bootstrapsize):
  np.random.seed(1227)
  boot = np.empty(bootstrapsize, dtype=float)
  # writing the mean value in the first sample
  boot[0] = np.mean(X)
  # doing all other samples
  for i in range(1, bootstrapsize):
    rnd = np.random.random_integers(0, high=len(X)-1, size=len(X))
    boot_dummy = 0.0 
    for j in range(0, len(X)):
      boot_dummy += X[rnd[j]]  
    boot[i] = boot_dummy/len(X)
  return boot

# symmetrising and bootstrapping
def sym_and_boot(X, T, nb_cfg, bootstrapsize = 1000, path = ''):
  boot = bootstrap(X[0:nb_cfg], bootstrapsize)
  for t in range(1, T/2):
    data = []
    for a, b in zip(X[t*nb_cfg:(t+1)*nb_cfg], X[(T-t)*nb_cfg:(T-t+1)*nb_cfg]):
      data.append( (a+b)/2.0)
    boot = np.c_[boot, bootstrap(data, bootstrapsize)]
  boot = np.c_[boot, bootstrap(X[(T/2)*nb_cfg:(T/2+1)*nb_cfg], bootstrapsize)]
  # writing bootstrapsample to file
  if len(path):
    IOcontraction.ensure_dir(path)
    np.save(path, boot)
  return boot

# ratio computation
def compute_ratio(C4, C2):
  print '\ncompute ratio:\n--------------\n' 
  ratio = np.empty([C4.shape[0], C4.shape[1]-1], dtype=float)
  for b in range(0, C4.shape[0]):
    rowC2 = C2[b,:]
    rowC4 = C4[b,:]
    for t in range(0, len(rowC4)-1):
      ratio[b, t] = (rowC4[t] - rowC4[t+1])/(rowC2[t]**2 - rowC2[t+1]**2)
  mean, err = mean_error_print(ratio)
  return ratio, mean, err

# derivative
def compute_derivative(boot):
#  print '\ncompute derivative:\n-------------------\n' 
  derv = np.empty((boot.shape[0], (boot.shape[1]-1)) + boot.shape[2:], dtype=float)
  # computing the derivative
  for b in range(0, boot.shape[0]):
    row = boot[b,:]
    for t in range(0, len(row)-3):
      derv[b, t] = -row[t+1] + row[t]
#      derv[b, t] = -row[-1] + row[t]
  mean, err = mean_error_print(derv)
  return derv, mean, err

# computes the mean and the error, and writes both out
def mean_error_print(boot, write = 0):
  mean = np.nanmean(boot, axis=0)
  err  = np.nanstd(boot, axis=0)
  if write:
    for t, m, e in zip(range(0, len(mean)), mean, err):
      print t, m, e
  return mean, err

# compute the mean correlator with error
def return_mean_corr(boot):
  print'\nmean correlator:\n----------------\n' 
  mean, err = mean_error_print(boot)
  return mean, err

# mass computation
def compute_mass(boot):
#  print '\ncompute mass:\n-------------\n' 
  # creating mass array from boot array
  mass = np.empty((boot.shape[0], boot.shape[1]-2) + boot.shape[2:], dtype=float)
  # computing the mass via formula
  for b, row in enumerate(boot): #range(0, boot.shape[0]):
#    row = boot[b,:]
    for t in range(1, len(row)-1):
      mass[b, t-1] = (row[t-1] + row[t+1])/(2.0*row[t])
  mass2 = np.arccosh(mass)
#  for m1, m2 in zip(mass, mass2):
#    print m1, m2
  mean, err = mean_error_print(mass2)
  return mass2, mean, err


