import numpy as np
import cmath

################################################################################
# setup of lookup tables for subduction coefficients. ##########################
################################################################################
# copy-pasted from Dudek et al.: "S and D-wave phase shifts in isospin-2 \pi\pi 
# scattering from lattice QCD" [43], \ Supplementary Material

#TODO: In Cms-case second momentum is the first's negative. Maybe do not
# hardcode as numpy-array

sqrt2=np.sqrt(2.)

################################################################################
# T1
T1 = [[[np.asarray([ 1,  0,  0], dtype=int),[np.asarray([-1,  0,  0], dtype=int), -1./2], \
       [np.asarray([ 0,  1,  0], dtype=int),[np.asarray([ 0, -1,  0], dtype=int),      -1j/2], \
       [np.asarray([ 0,  0,  1], dtype=int),[np.asarray([ 0,  0, -1], dtype=int),  0.], \
       [np.asarray([-1,  0,  0], dtype=int),[np.asarray([ 1,  0,  0], dtype=int),  1./2], \
       [np.asarray([ 0, -1,  0], dtype=int),[np.asarray([ 0,  1,  0], dtype=int),       1j/2], \
       [np.asarray([ 0,  0, -1], dtype=int),[np.asarray([ 0,  0,  1], dtype=int),  0.], \
       [np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1, -1,  0], dtype=int), -1./4-1j/4], \
       [np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0, -1, -1], dtype=int),      -1j/4], \
       [np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  0, -1], dtype=int), -1./4], \
       [np.asarray([ 1, -1,  0], dtype=int), np.asarray([-1,  1,  0], dtype=int), -1./4+1j/4], \
       [np.asarray([ 0,  1, -1], dtype=int), np.asarray([ 0, -1,  1], dtype=int),      -1j/4], \
       [np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  0, -1], dtype=int),  1./4], \
       [np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1, -1,  0], dtype=int),  1./4-1j/4], \
       [np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  1, -1], dtype=int),       1j/4], \
       [np.asarray([ 1,  0, -1], dtype=int), np.asarray([-1,  0,  1], dtype=int), -1./4], \
       [np.asarray([-1, -1,  0], dtype=int), np.asarray([ 1,  1,  0], dtype=int),  1./4+1j/4], \
       [np.asarray([ 0, -1, -1], dtype=int), np.asarray([ 0,  1,  1], dtype=int),       1j/4], \
       [np.asarray([-1,  0, -1], dtype=int), np.asarray([ 1,  0,  1], dtype=int),  1./4]]

      [[np.asarray([ 1,  0,  0], dtype=int),[np.asarray([-1,  0,  0], dtype=int),  0.], \
       [np.asarray([ 0,  1,  0], dtype=int),[np.asarray([ 0, -1,  0], dtype=int),  0.], \
       [np.asarray([ 0,  0,  1], dtype=int),[np.asarray([ 0,  0, -1], dtype=int),  1./sqrt2], \
       [np.asarray([-1,  0,  0], dtype=int),[np.asarray([ 1,  0,  0], dtype=int),  0.], \
       [np.asarray([ 0, -1,  0], dtype=int),[np.asarray([ 0,  1,  0], dtype=int),  0.], \
       [np.asarray([ 0,  0, -1], dtype=int),[np.asarray([ 0,  0,  1], dtype=int), -1./sqrt2], \
       [np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1, -1,  0], dtype=int),  0.], \
       [np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0, -1, -1], dtype=int),  1./(2*sqrt2)], \
       [np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  0, -1], dtype=int),  1./(2*sqrt2)], \
       [np.asarray([ 1, -1,  0], dtype=int), np.asarray([-1,  1,  0], dtype=int),  0.], \
       [np.asarray([ 0,  1, -1], dtype=int), np.asarray([ 0, -1,  1], dtype=int), -1./(2*sqrt2)], \
       [np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  0, -1], dtype=int),  1./(2*sqrt2)], \
       [np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1, -1,  0], dtype=int),  0.], \
       [np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  1, -1], dtype=int),  1./(2*sqrt2)], \
       [np.asarray([ 1,  0, -1], dtype=int), np.asarray([-1,  0,  1], dtype=int), -1./(2*sqrt2)], \
       [np.asarray([-1, -1,  0], dtype=int), np.asarray([ 1,  1,  0], dtype=int),  0.], \
       [np.asarray([ 0, -1, -1], dtype=int), np.asarray([ 0,  1,  1], dtype=int), -1./(2*sqrt2)], \
       [np.asarray([-1,  0, -1], dtype=int), np.asarray([ 1,  0,  1], dtype=int), -1./(2*sqrt2)]]

      [[np.asarray([ 1,  0,  0], dtype=int),[np.asarray([-1,  0,  0], dtype=int),  1./2], \
       [np.asarray([ 0,  1,  0], dtype=int),[np.asarray([ 0, -1,  0], dtype=int),      -1j/2], \
       [np.asarray([ 0,  0,  1], dtype=int),[np.asarray([ 0,  0, -1], dtype=int),  0.], \
       [np.asarray([-1,  0,  0], dtype=int),[np.asarray([ 1,  0,  0], dtype=int), -1./2], \
       [np.asarray([ 0, -1,  0], dtype=int),[np.asarray([ 0,  1,  0], dtype=int),       1j/2], \
       [np.asarray([ 0,  0, -1], dtype=int),[np.asarray([ 0,  0,  1], dtype=int),  0.], \
       [np.asarray([ 1,  1,  0], dtype=int), np.asarray([-1, -1,  0], dtype=int),  1./4-1j/4], \
       [np.asarray([ 0,  1,  1], dtype=int), np.asarray([ 0, -1, -1], dtype=int),      -1j/4], \
       [np.asarray([ 1,  0,  1], dtype=int), np.asarray([-1,  0, -1], dtype=int),  1./4], \
       [np.asarray([ 1, -1,  0], dtype=int), np.asarray([-1,  1,  0], dtype=int),  1./4+1j/4], \
       [np.asarray([ 0,  1, -1], dtype=int), np.asarray([ 0, -1,  1], dtype=int),      -1j/4], \
       [np.asarray([-1,  0,  1], dtype=int), np.asarray([ 1,  0, -1], dtype=int), -1./4], \
       [np.asarray([-1,  1,  0], dtype=int), np.asarray([ 1, -1,  0], dtype=int), -1./4-1j/4], \
       [np.asarray([ 0, -1,  1], dtype=int), np.asarray([ 0,  1, -1], dtype=int),       1j/4], \
       [np.asarray([ 1,  0, -1], dtype=int), np.asarray([-1,  0,  1], dtype=int),  1./4], \
       [np.asarray([-1, -1,  0], dtype=int), np.asarray([ 1,  1,  0], dtype=int), -1./4+1j/4], \
       [np.asarray([ 0, -1, -1], dtype=int), np.asarray([ 0,  1,  1], dtype=int),       1j/4], \
       [np.asarray([-1,  0, -1], dtype=int), np.asarray([ 1,  0,  1], dtype=int), -1./4]]]

################################################################################
def coefficients(irrep):
  if irrep is 'T1':
    return T1

