################################################################################
#
# Author: Bastian Knippschild (B.Knippschild@gmx.de) 
# Date:   October 2014 
#
# Copyright (C) 2014 Bastian Knippschild
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################
#
# Function: Computation of Luescher's Zeta Function. This program is based on 
#           arXiv:1107.5023v2, e.g. equation (5). The parameter \Lambda is set 
#           to 1 in this implementation. 
#
# For informations on input parameters see the description of the function.
#
################################################################################
#
# Performed tests:
# 1.) Against Mathematica code provided by Liuming Liu w. and w.o. tbc in cms
#     and l=0, m=0 
# 2.) Against data from arXiv:1107.5023v2 w. and w.o. moving frames and l=0, m=0
# 3.) Against data from arXiv:1011.5288 w. and w.0. moving frames and linear 
#     combinations of l=2, m=-2,0,2. 
#
#  See the test function at the very end of this file for more information!
#
################################################################################

import os
import math
import cmath
import numpy as np
import scipy.special
import scipy.integrate

from ._zeta_memoize import memoize
#import create_momentum_array as cma
from .create_momentum_array import main as cmamain

def zeta_n(function):
    path = "./momenta.npy"
    if not os.path.isfile(path):
        #import .create_momentum_array as cma
        cmamain()
    try:
        _mem = np.load(path)
        #print("reading n in wrapper")
    except (IOError, UnicodeDecodeError):
        #import .create_momentum_array as cma
        cmamain()
        _mem = np.load(path)
        #print("error and reading n in wrapper")

    def zeta_wrapper(*args, **kwargs):
        if 'n' in kwargs:
            kwargs.pop('n')
        return function(*args, n=_mem, **kwargs)

    zeta_wrapper._zeta_n_mem = _mem
    zeta_wrapper._zeta_n_origfunc = function
    zeta_wrapper._zeta_n_func_name = function.func_name
    return zeta_wrapper

################################################################################
#
#                            Luescher's Zeta Function
#
# This is the ONLY function which should and needs to be called from outside.
#
# input: q2       : (IMPORTANT:) SQUARED scattering momentum fraction, ONLY 
#                   MANDATORY INPUT PARAMETER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#        gamma    : Lorentz factor for moving frames, see e.g. arXiv:1011.5288
#        l        : orbital quantum number
#        m        : magnetic quantum number
#        d        : total three momentum of the system. (TBC: d can be used as 
#                   a twist angle as well. The correspondance is:
#                          d = -theta/pi     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#        m_split  : coefficient when the masses of the scattering particles are
#                   different. It is given by: m_split = 1+(m_1^2-m_2^2)/E_cm^2
#                   where E_cm^2 is the interacting energy in the cms.
#        precision: precision of the calculation
#        verbose  : 0, no output on screen; 1, detailed output with convergence
#                   informations 
#        n        : Do not use, used for wrapping the function.
#
# return: The value of Luescher's Zeta function as a COMPLEX number.
#
# minor details: The three terms A, B, and C correspond to the three terms of
#                equation (5) in arXiv:1107.5023v2.
#
################################################################################
@zeta_n
def Z(q2, gamma = 1.0, l = 0, m = 0, d = np.array([0., 0., 0.]), \
      m_split = 1, precision = 10e-6, verbose = 0, n=None):
  # some small checks
  if gamma < 1.0:
    print( 'Gamma must be larger or equal to 1.0')
    exit(0)
  if n==None:
      # reading the three momenta for summation from file
      print("loading n zeta")
      _n = np.load("./momenta.npy")
  else:
      _n = n
  # the computation
  _a = A(q2, gamma, l, m, d, precision, verbose, m_split, _n)
  _b = B(q2, gamma, l, precision, verbose)
  _c = C(q2, gamma, l, m, d, precision, verbose, m_split, _n)
  #print("firstPart  = %.24f + %+.24fI" % (_a.real, _a.imag))
  #print("secondPart = %.24f + %+.24fI" % (_b.real, _b.imag))
  #print("thirdPart  = %.24f + %+.24fI" % (_c.real, _c.imag))
  res = _a + _b + _c
  if verbose:
    print( 'Luescher Zeta function:', res)
  return res

################################################################################


################################################################################
#
#                            IMPLEMENTATION
#
################################################################################

# Transforms an array of 3d vectors from cartesian to spherical coordinates
################################################################################
def appendSpherical_np(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) 
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

# Computes the vector r for the sum in term A and returns it in spherical 
# coordinates 
################################################################################
def compute_r_in_spherical_coordinates(a, d, gamma, m_split):
  out = np.zeros_like(a)
  if (np.linalg.norm(d) == 0.0):
    out = a/gamma
  # splitting every vector in a in parallel and orthogonal part w.r.t. d
  else:
    for i, r in enumerate(a):
      r_p = np.dot(r, d)/np.dot(d,d)*d 
      r_o = r-r_p
      out[i,:] = (r_p-0.5*m_split*d)/gamma + r_o
  return appendSpherical_np(out)

# compute spherical harmonics
################################################################################
def sph_harm(m = 0, l = 0, phi = 0, theta = 0):
  if l is 0 and m is 0:
    return 0.28209479177387814
  elif l is 1 and m is -1:
    return 0.3454941494713355*np.sin(theta)*np.exp(-1.j*phi)
  elif l is 1 and m is  0:
    return 0.48860251190291992*np.cos(theta)
  elif l is 1 and m is  1:
    return -0.3454941494713355*np.sin(theta)*np.exp(1.j*phi)
  elif l is 2 and m is -2:
    return 0.38627420202318957*np.sin(theta)*np.sin(theta) * np.exp(-2.*1.j*phi)
  elif l is 2 and m is  0:
    return 0.31539156525252005*(3.*np.cos(theta)*np.cos(theta) - 1.)
  elif l is 2 and m is  2:
    return 0.38627420202318957*np.sin(theta)*np.sin(theta) * np.exp(2.*1.j*phi)
  elif l is 4 and m is -4:
    return 0.44253269244498263*np.sin(theta)**4*np.exp(-4.*1.j*phi)
  elif l is 4 and m is -2:
    return 0.33452327177864458*\
        np.sin(theta)**2.*(7.*np.cos(theta)**2. - 1.)*np.exp(-2.*1.j*phi)
  elif l is 4 and m is  0:
    return 0.10578554691520430*(35.*np.cos(theta)**4 - 30.*np.cos(theta)**2 +3.)
  elif l is 4 and m is  2:
    return 0.33452327177864458*\
        np.sin(theta)**2.*(7.*np.cos(theta)**2. - 1.)*np.exp(2.*1.j*phi)
  elif l is 4 and m is  4:
    return 0.44253269244498263*np.sin(theta)**4*np.exp(4.*1.j*phi)
  else:
    return scipy.special.sph_harm(m, l, phi, theta)

# returns the part of the momentum array for a given momentum squared
################################################################################
def return_momentum_array(p, n):
  if len(n[p,0]) is 0:
    p += 1
  out = n[p, 0]
  p += 1
  return out, p

# Computes a part of the sum in term A
################################################################################
def compute_summands_A(a_sph, q, l, m): 
  inter = []
  for count, r in enumerate(a_sph):
    breaker = 0
    # one way to avoid double computation
    for i in xrange(0, count):
      if abs(float(r[0]) - float(a_sph[i, 0])) < 1e-8:
        inter.append(inter[i])
        breaker = 1
        break
    if breaker == 0:
      inter.append((np.exp(-(r[0]**2.-q)) * r[0]**l) / (r[0]**2-q))

  result = 0.0 
  for count, r in enumerate(a_sph):
    result += inter[count]*sph_harm(m, l, r[2], r[1]) 
  return result

# Computation of term A
################################################################################
def A(q, gamma, l, m, d, precision, verbose, m_split, n):
  i = 0
  r, i = return_momentum_array(i, n)
  r_sph = compute_r_in_spherical_coordinates(r, d, gamma, m_split)
  result = compute_summands_A(r_sph, q, l, m)
  if verbose:
    print('convergence in term A:')
    print('\t', i-1, result)
  # computing new sums until precision is reached
  eps = 1
  while (eps > precision):
    r, i = return_momentum_array(i, n)
    r_sph = compute_r_in_spherical_coordinates(r, d, gamma, m_split)
    result_h = compute_summands_A(r_sph, q, l, m)
    if result != 0.0:
      eps = abs(result_h/result)
    result += result_h
    if verbose:
      print('\t', i-1, result, eps)
    # if result is still zero after 4 iterations it is assumed to stay zero
    if result == 0.0 and i > 4:
      break
      print('\t', i-1, result, eps)
  if verbose:
    print('Term A:', result)
  return result

# Computation of term B
################################################################################
#@memoize(5000)
def B(q, gamma, l, precision, verbose):
  if l is not 0:
    return 0.0 
  else:
    a = 2.*scipy.special.sph_harm(0, 0, 0.0, 0.0)*gamma*math.pow(math.pi, 3./2.)
    # The integral gives [2*(exp(q)*DawsonF(sqrt(q))/sqrt(q)] for Lambda = 1
    # The Dawson function is only available in scipy 0.13 or so, so it is
    # replaced by a representation with the Gaussian error function.
    dawson = -1.j*np.sqrt(math.pi) * np.exp(-q) * \
             scipy.special.erf(1.j*cmath.sqrt(q)) / 2.
    b = q * 2.*np.exp(q)*dawson/cmath.sqrt(q)
    c = math.exp(q)
    if verbose:
      print('Term B:', a*(b-c))
    return a*(b-c)

# Computes the term gamma*w and returns the result in spherical coordinates
################################################################################
def compute_gamma_w_in_spherical_coordinates(a, d, gamma):
  out = np.zeros(a.shape)
  if (np.linalg.norm(d) == 0.0):
    for r, i in zip(a, xrange(0,a.shape[0])):
      out[i,:] = r*gamma
  # splitting every vector in a in parallel and orthogonal part w.r.t. d
  else:
    for r, i in zip(a, xrange(0,a.shape[0])):
      r_p = np.dot(r, d)/np.dot(d,d)*d 
      r_o = r-r_p
      out[i,:] = r_p*gamma + r_o
  return appendSpherical_np(out)

# Just the integrand of term C
################################################################################
integrand = lambda t, q, l, w: ((math.pi/t)**(3./2.+l) ) * np.exp(q*t-w/t)

# Computes a part of the sum in term C
################################################################################
def compute_summands_C(w_sph, w, q, gamma, l, m, d, m_split, precision):
  part1 = (-1.j)**l * gamma * (np.absolute(w_sph[:,0])**l) * \
          np.exp((-1.j)*m_split*math.pi*np.dot(w, d)) * \
          sph_harm(m, l, w_sph[:,2], w_sph[:,1])
  # Factor two: The integral 
  part2 = []
  for count, ww in enumerate(w_sph):
    breaker = 0
    for i in xrange(0, count):
      if abs(float(ww[0]) - float(w_sph[i, 0])) < 1e-8:
        part2.append(part2[i])
        breaker = 1
        break
    if breaker == 0:
      # the precision in this integral might be crucial at some point but it is
      # rather high right now with a standard of 1e-8. It should be enough
      # for all comoputations. In doubt, please change it.
      part2.append((scipy.integrate.quadrature(integrand, 0., 1., \
                    args=(q, l, (math.pi*ww[0])**2), tol = precision*0.1, \
                    maxiter=1000))[0])
  part2 = np.asarray(part2, dtype=float)
  # return the result
  return np.dot(part1, part2)

# Computation of term C
################################################################################
def C(q, gamma, l, m, d, precision, verbose, m_split, n):
  i = 1
  w, i = return_momentum_array(i, n)
  w_sph = compute_gamma_w_in_spherical_coordinates(w, d, gamma)
  result = compute_summands_C(w_sph, w, q, gamma, l, m, d, m_split, precision)
  if verbose:
    print('convergence in term C:')
    print('\t', i-1, result)
  # computing new sums until precision is reached
  eps = 1
  while (eps > precision):
    w, i = return_momentum_array(i, n)
    w_sph = compute_gamma_w_in_spherical_coordinates(w, d, gamma)
    result_h = compute_summands_C(w_sph, w, q, gamma, l, m, d, m_split, precision)
    if result != 0.0:
      eps = abs(result_h/result)
    result += result_h
    if verbose:
      print('\t', i-1, result, eps)
    # if result is still zero after 4 iterations it is assumed to stay zero
    if result == 0.0 and i > 4:
      break
  if verbose:
    print('Term C:', result)
  return result


def test(): 
  # cms ##########################
  print('\nTest in cms:')
  Pcm = np.array([0., 0., 0.])
  q = 0.1207*24/(2.*math.pi)
  gamma = 1.0
  zeta = Z(q*q, gamma, d = Pcm).real
  print('q, gamma:', q, gamma)
  delta = np.arctan(math.pi**(3./2.)*q/zeta)*180./math.pi
  if delta < 0:
    delta = 180+delta
  print('delta:', delta, 'delta should be: 136.6527')
  
  # mv1 ##########################
  print('\nTest in mv1:')
  Pcm = np.array([0., 0., 1.])
  L = 32
  q = 0.161*L/(2.*math.pi)
  E = 0.440
  Ecm = 0.396
  gamma = E/Ecm
  Z00 = Z(q*q, gamma, d = Pcm).real
  Z20 = Z(q*q, gamma, d = Pcm, l = 2).real
  print('q, gamma:', q, gamma)
  delta = np.arctan(gamma*math.pi**(3./2.) * q / \
          (Z00 + (2./(q*q*math.sqrt(5)))*Z20))*180./math.pi
  if delta < 0:
    delta = 180+delta
  print('delta:', delta, 'delta should be: 115.7653')
  
  
  # mv2 ##########################
  print('\nTest in mv2:')
  Pcm = np.array([1., 1., 0.])
  L = 32
  q = 0.167*L/(2.*math.pi)
  E = 0.490
  Ecm = 0.407
  gamma = E/Ecm
  Z00 = Z(q*q, gamma, d = Pcm).real
  Z20 = Z(q*q, gamma, d = Pcm, l = 2).real
  Z22  = Z(q*q, gamma, d = Pcm, l = 2, m = 2).imag
  Z2_2 = Z(q*q, gamma, d = Pcm, l = 2, m = -2).imag
  print('q, gamma:', q, gamma)
  delta = np.arctan(gamma*math.pi**(3./2.) * q / \
          (Z00 - (1./(q*q*math.sqrt(5)))*Z20 \
          + ((math.sqrt(3./10.)/(q*q))*(Z22-Z2_2))))*180./math.pi
  if delta < 0:
    delta = 180+delta
  print('delta:', delta, 'delta should be: 127.9930')

if __name__ == '__main__':
  test()
