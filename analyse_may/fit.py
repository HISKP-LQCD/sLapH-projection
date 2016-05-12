#!/usr/bin/python

from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import scipy.stats
import numpy as np

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab


###############################################################################
# Fit routine
def fitting(fitfunc, X, Y, start_parm,  correlated = 1, tcut = 0., \
            writescreen = 1, deltaE = np.empty(1)):
  errfunc = lambda p, x, y, error, tc: np.dot(error, (y-fitfunc(p,x,tc)).T)
  # compute inverse, cholesky decomposed covariance matrix
  if correlated == 0:
    if writescreen:
      print '\tPerforming an uncorrelated fit!'
    cov = np.diag(np.diagonal(np.cov(Y.T)))
  else:
    if writescreen:
      print '\tPerforming a correlated fit!'
    cov = np.cov(Y.T)
  cov = (np.linalg.cholesky(np.linalg.inv(cov))).T

  # degrees of freedom
  dof = float(Y.shape[1]-len(start_parm)) 
  # The FIT to the boostrap samples
  res = np.zeros((Y.shape[0], len(start_parm)))
  chisquare = np.zeros(Y.shape[0])
  for b in range(0, Y.shape[0]):
    p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm, \
                             args=(X, Y[b,:], cov, deltaE[b]), full_output=1, \
                               factor=0.1)
    chisquare[b] = float(sum(infodict['fvec']**2.))
    res[b] = np.array(p)
  res_mean, res_std = res[0], np.std(res, axis=0)
  # p-value
  chi2 = chisquare[0]
  pvalue = 1.-scipy.stats.chi2.cdf(chi2, dof)

  # The fit to the mean value
  y = np.mean(Y, axis=0)
  p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm, \
                             args=(X, y, cov, tcut), full_output=1)

  # writing results to screen
  if writescreen:
    print '\tDegrees of freedom:', dof
    print '\n\tFit results from bootstrap fit:'
    for rm, rs in zip(res_mean, res_std):
      print '\t%.6e' % rm, '+/-', '%.6e' % rs
    print '\tChi^2/dof: %.6e' % (chi2/dof), '+/- %.6e' % (np.std(chisquare)/dof)
    print '\tp-value: ', pvalue 
    print '\n\tFit parameter from mean value fit:'
    for pp in p:
      print '\t%.6e' % pp
    print '\tChi^2/dof: %.6e' % float(sum(infodict['fvec']**2.)/dof)

  return res, chi2, pvalue



###############################################################################
# Fit routine with explicit derivative
def fitting_w_derv(fitfunc, dfunc, X, Y, start_parm,  correlated = 1, tcut = 0., \
            writescreen = 1):
  errfunc = lambda p, x, y, error, tc: np.dot(error, (y-fitfunc(p,x,tc).T))
  # compute inverse, cholesky decomposed covariance matrix
  if correlated == 0:
    if writescreen:
      print '\tPerforming an uncorrelated fit!'
    cov = np.diag(np.diagonal(np.cov(Y.T)))
  else:
    if writescreen:
      print '\tPerforming a correlated fit!'
    cov = np.cov(Y.T)
  cov = (np.linalg.cholesky(np.linalg.inv(cov))).T

  # degrees of freedom
  dof = float(Y.shape[1]-len(start_parm)) 
  # The FIT to the boostrap samples
  res = np.zeros((Y.shape[0], len(start_parm)))
  chisquare = np.zeros(Y.shape[0])
  for b in range(0, Y.shape[0]):
    p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm, \
                               args=(X, Y[b,:], cov, tcut), Dfun=dfunc, \
                               col_deriv=1, full_output=1, factor=0.1)
    chisquare[b] = float(sum(infodict['fvec']**2.))
    res[b] = np.array(p)
  res_mean, res_std = np.mean(res, axis=0), np.std(res, axis=0)
  # p-value
  chi2 = np.median(chisquare)
  pvalue = 1.-scipy.stats.chi2.cdf(chi2, dof)

  # The fit to the mean value
  y = np.mean(Y, axis=0)
  p,cov1,infodict,mesg,ier = leastsq(errfunc, start_parm, \
                             args=(X, y, cov, tcut), full_output=1)

  # writing results to screen
  if writescreen:
    print '\tDegrees of freedom:', dof
    print '\n\tFit results from bootstrap fit:'
    for rm, rs in zip(res_mean, res_std):
      print '\t%.6e' % rm, '+/-', '%.6e' % rs
    print '\tChi^2/dof: %.6e' % (chi2/dof), '+/- %.6e' % (np.std(chisquare)/dof)
    print '\tp-value: ', pvalue 
    print '\n\tFit parameter from mean value fit:'
    for pp in p:
      print '\t%.6e' % pp
    print '\tChi^2/dof: %.6e' % float(sum(infodict['fvec']**2.)/dof)

  return res, chi2, pvalue
