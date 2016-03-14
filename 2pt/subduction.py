#!/usr/bin/python

import numpy as np
import cmath

import sys
#sys.stdout = open('./subduction.out', 'w')

import irreps as representation
import utils as utils

################################################################################
# parameters ###################################################################

################################################################################
p = 2         # momentum

verbose = 0

# Operators entering the Gevp. Last entry must contain the name in LaTeX 
# compatible notation for plot labels
gamma_i =   [1, 2, 3, '\gamma_i']
gamma_0i =  [10, 11, 12, '\gamma_0\gamma_i']
gamma_50i = [15, 14, 13, '\gamma_5\gamma_0\gamma_i']

gamma = [gamma_i, gamma_0i, gamma_50i]
#gamma_for_filenames = ['gi', 'g0gi', 'g5g0gi']
gamma_for_filenames = {'\gamma_i' : 'gi', '\gamma_0\gamma_i' : 'g0gi', '\gamma_5\gamma_0\gamma_i' : 'g5g0gi'}

################################################################################
# Subduction Routine ###########################################################

################################################################################
# read original data and quantum numbers
# data must have shape [operators x ...]

def subduce_ensembles(p, gamma, gamma_for_filenames, verbose=0):
  filename = './readdata/p%1i/C20_p%1i.npy' % (p, p)
  data = np.load(filename)
  if verbose:
    print data.shape
  
  path = './readdata/p%1i/C20_p%1i_quantum_numbers.npy' % (p, p)
  quantum_numbers = np.load(path)
  
  if ( (quantum_numbers.shape[0] != data.shape[0])):
    print '\tRead correlators do not aggree with quantum numbers'
    exit(0)
  
  ################################################################################
  # From analytic calculation: Irreps contributing to momenta 0 <= p <= 4
  if p in [0]:
    irreps = [['T1']]
  elif p in [1, 3, 4]:
    irreps = [['A1', 'E2_1', 'E2_2']]
  elif p in [2]:
    irreps = [['A1', 'B1', 'B2']]
  
  # get factors for the desired irreps
  for i in irreps[-1]:
    irreps.insert(-1,representation.coefficients(i))
  
  ################################################################################
  # loop over all momenta and add the correlators with correct subduction 
  # coefficients if the operators are correct
  # TODO: alert when necessary operator for subduction missing in data
  # TODO: catch dublicates in quantum numbers -> done in read-py?
  corr_irreps = []
  quantum_numbers_irreps = []
  for i, irrep in enumerate(irreps[:-1]):
    corr_gamma = []
    quantum_numbers_gamma = []
    for g1 in gamma:
      for g2 in gamma:
        corr = np.zeros((0, ) + data[0].shape, dtype=np.double)
        # second dimension: three-momentum + name
        quantum_numbers_irrep = np.zeros((0,6))
        for el in irrep:
          subduced_correlator = np.zeros((1,) + data[0].shape)
          for op, qn in enumerate(quantum_numbers):
            # correlator in all dimensions but operator and Re/Im. In this case T and
            # nb_boot
            if np.array_equal(el[0], qn[0]):
              for g_so in range(0,3):
                for g_si in range(0,3):
                  # Hardcoded 2pt functions here because 
                  #   sink operator = conj(source operator)
                  # and irreps are diagonal. Otherwise el[g_si] must be calculated in 
                  # seperate loop over the sink momentum
                  factor = el[g_so+1] * np.conj(el[g_si+1])
                  if (factor == 0):
                    continue
                  # old signs for non-hermitian operators
#                  # account for extra - sign in definition of g_5g_0g_i compared 
#                  # to g_1g_3. "!=" is a xor
#                  if (g1[g_so] == 14) != (g2[g_si] == 14):
#                    factor = (-1) * factor
#                  # account for additional i from twisting \tau_3 to \mathbb{1} 
#                  # in g_5g_0g_i
#                  if (g1[g_so] in gamma_50i):
#                    factor = (1j) * factor
#                  if (g2[g_si] in gamma_50i):
#                    factor = (1j) * factor
  

                  # calculating the Correlators yields imaginary parts in 
                  # gi-g0gi, gi-gjgk and cc. Switching real and imaginary
                  # part in the subduction factor accounts for this.
                  if (g1[g_so] in gamma_i) != (g2[g_si] in gamma_i):
                    factor = factor.imag + factor.real * 1j
#                    factor = (1j) * factor
  
                  if ( (g1[g_so] == qn[2]) and (g2[g_si] == qn[5]) ):
                    subduced_correlator[0] = subduced_correlator[0] + \
                             factor.real * data[op].real + factor.imag * data[op].imag
                    if verbose:
                      print '\tsubduced g_so = %i, g_si = %i' % (g1[g_so], g2[g_si])
                      print '\t\tsubduction coefficient = % .2f + % .2fi' % \
                                                            (factor.real, factor.imag)
        #          else:
        #            print 'Could not find operator with p = %s, g_so = %i, g_si = %i' \
        #                  ' needed for subduction into A1 irrep!' % (a1[0], g_so, g_si)
        #            exit(0)
          # if momentum from A1 lookup table belongs to desired irrep
          if(subduced_correlator.any() != 0):
            if verbose:
              print '\tinto momentum (%i,%i,%i)' % (el[0][0], el[0][1], el[0][2])
              print ' '
            corr = np.vstack((corr, subduced_correlator))
            quantum_numbers_irrep = np.vstack((quantum_numbers_irrep, \
                            np.hstack((el[0], g1[-1], g2[-1], irreps[-1][i])) ))
        corr_gamma.append(corr)
        quantum_numbers_gamma.append(quantum_numbers_irrep)
  
    corr_irreps.append(np.asarray(corr_gamma))
    quantum_numbers_irreps.append(np.asarray(quantum_numbers_gamma))
  corr_irreps = np.asarray(corr_irreps)
  print corr_irreps.shape
  quantum_numbers_irreps = np.asarray(quantum_numbers_irreps)
  
  ################################################################################
  # write data to disc
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i/' % p)
  #utils.ensure_dir('./readdata/p%1i/single' % p)
  utils.ensure_dir('./binarydata/')
  utils.ensure_dir('./binarydata/p%1i/' % p)
  
  # write all subduced correlators
  path = './readdata/p%1i/C20_p%1i_single_subduced' % (p, p)
  np.save(path, corr_irreps)
  path = './readdata/p%1i/C20_p%1i_single_subduced_quantum_numbers' % (p, p)
  np.save(path, quantum_numbers_irreps)
  
  # write all subduced and averaged correlators
  # average is always over subduced operators contributing to same irrep
  # TODO: change that to 'E2_1' in irreps[-1]
  if p not in [1,3,4]:
    path = './readdata/p%1i/C20_p%1i_avg_subduced' % (p, p)
    np.save(path, np.mean(corr_irreps, axis=1))
    path = './readdata/p%1i/C20_p%1i_avg_subduced_quantum_numbers' % (p, p)
    np.save(path, quantum_numbers_irreps[...,0,3])
  else:
    # I hate the E2 irrep. 
    # Average E2_1 and E2_2 additionally and append the result
    path = './readdata/p%1i/C20_p%1i_avg_subduced' % (p, p)
    E2 = np.zeros((1,) + np.mean(corr_irreps, axis=2)[0].shape)
    for j in range(0,corr_irreps.shape[1]):
      E2[0, j] = np.mean(np.vstack((corr_irreps[1,j], corr_irreps[2,j])), axis=0)
    np.save(path, np.vstack((np.mean(corr_irreps, axis=2), E2 )) )
    path = './readdata/p%1i/C20_p%1i_avg_subduced_quantum_numbers' % (p, p)
    E2_qn = np.zeros((1,quantum_numbers_irreps.shape[1], \
                      quantum_numbers_irreps[...,-3:].shape[-1]),dtype=((str,256)))
    E2_qn[0] = quantum_numbers_irreps[0,:,0,-3:]
    for j in range(0,corr_irreps.shape[1]):
      E2_qn[0,j,-1] = 'E2'
    np.save(path, np.vstack((quantum_numbers_irreps[...,0,-3:], E2_qn)) )
  
  ################################################################################
  # write all subduced correlators for each irrep and gamma seperately in ascii 
  # format
  if(corr_irreps.shape[0] != (len(irreps)-1) ):
    print '\tThe number of calculated irreps is not the number of existing ' \
          'ones!'
    exit(0)
  if(corr_irreps.shape[1] != len(gamma)*len(gamma) ):
    print '\tThe number of calculated gamma combinations is not the number of ' \
          'existing ones!'
    exit(0)
  
  #TODO: finish output to binary files with dictionary
  #TODO: somehow make this dictionary more elegant
  if p not in [1,3,4]:
    nb_irreps = corr_irreps.shape[0]
  else:
    # Average E2_1 and E2_2 additionally and append the result
    for j in ( range(0, corr_irreps.shape[1]) ):
      path = './binarydata/p%1i/C20_p%1i_E2_%s_%s.dat' % (p, p, \
                        gamma_for_filenames[quantum_numbers_irreps[i,j,0,-3]], \
                        gamma_for_filenames[quantum_numbers_irreps[i,j,0,-2]])
      f = open(path, 'wb')
      E2 = np.mean(np.vstack((corr_irreps[-1,j], corr_irreps[-2,j])), axis=0)
      (E2.swapaxes(-1, -2)).tofile(f)
      f.close()
    nb_irreps = corr_irreps.shape[0]-2

  for i in ( range(0, nb_irreps) ):
    for j in ( range(0, corr_irreps.shape[1]) ):
      # write in format p_irrep_gso_gsi
      path = './binarydata/p%1i/C20_p%1i_%s_%s_%s.dat' % (p, p, \
                        quantum_numbers_irreps[i,j,0,-1], \
                        gamma_for_filenames[quantum_numbers_irreps[i,j,0,-3]], \
                        gamma_for_filenames[quantum_numbers_irreps[i,j,0,-2]])
      f = open(path, 'wb')
      (corr_irreps[i,j].swapaxes(-1, -2)).tofile(f)
      f.close()
 
  ################################################################################
  # write all subduced correlators for each irrep and gamma seperately
  #if(corr_irreps.shape[0] != (len(irreps)-1) ):
  #  print '\tThe number of calculated irreps is not the number of existing ' \
  #        'ones!'
  #  exit(0)
  #if(corr_irreps.shape[1] != len(gamma)*len(gamma) ):
  #  print '\tThe number of calculated gamma combinations is not the number of ' \
  #        'existing ones!'
  #  exit(0)
  #
  #for i in ( range(0, corr_irreps.shape[0]) ):
  #  for j in ( range(0, corr_irreps.shape[1]) ):
  #   path = './readdata/p%1i/single/C20_p%1i_single_%s_%s_%s' % (p, p, \
  #           quantum_numbers_irreps[i,j,0,-3], quantum_numbers_irreps[i,j,0,-2], \
  #                                               quantum_numbers_irreps[i,j,0,-1])
  #   np.save(path, corr_irreps[i])
  #   path = './readdata/p%1i/single/C20_p%1i_single_%s_%s_%squantum_numbers' % (p, p, \
  #           quantum_numbers_irreps[i,j,0,-3], quantum_numbers_irreps[i,j,0,-2], \
  #                                               quantum_numbers_irreps[i,j,0,-1])
  #   np.save(path, quantum_numbers_irreps[i])

subduce_ensembles(p, gamma, gamma_for_filenames, verbose=0)


