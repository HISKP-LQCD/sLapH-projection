#!/usr/bin/python

import numpy as np
import cmath

import sys
#sys.stdout = open('./subduction.out', 'w')

import clebsch_gordan as cg
import utils as utils

################################################################################
# parameters ###################################################################

################################################################################
p = range(5)         # momentum

verbose = 0

# Operators entering the Gevp. Last entry must contain the name in LaTeX 
# compatible notation for plot labels
gamma_i =   [1, 2, 3, 'gi']
gamma_0i =  [10, 11, 12, 'g0gi']
gamma_50i = [15, 14, 13, 'g5g0gi']

gamma = [gamma_i, gamma_0i, gamma_50i]

################################################################################
# Subduction Routine ###########################################################

################################################################################
# read original data and quantum numbers
# data must have shape [operators x ...]

def subduce_ensembles(p_cm, gamma, verbose=0):
  print 'subducing p = %i' % p_cm

  filename = './readdata/p%1i/C20_p%1i.npy' % (p_cm, p_cm)
  data = np.load(filename)
  if verbose:
    print data.shape
  
  path = './readdata/p%1i/C20_p%1i_quantum_numbers.npy' % (p_cm, p_cm)
  qn_data = np.load(path)
  
  if ( (qn_data.shape[0] != data.shape[0])):
    print '\tRead correlators do not aggree with quantum numbers'
    exit(0)
  
  ##############################################################################
  # From analytic calculation: Irreps contributing to momenta 0 <= p <= 4
  if p_cm in [0]:
    irreps = [['T1']]
  elif p_cm in [1, 3, 4]:
    irreps = [['A1', 'E2']]
  elif p_cm in [2]:
    irreps = [['A1', 'B1', 'B2']]
  
  # get factors for the desired irreps
  for i in irreps[-1]:
    irreps.insert(-1,cg.coefficients(i))
  
  ##############################################################################
  # loop over all momenta and add the correlators with correct subduction 
  # coefficients if the operators are correct
  # TODO: alert when necessary operator for subduction missing in data
  # TODO: catch dublicates in quantum numbers -> done in read-py?
  correlator = []
  qn_subduced = []

  # loop over irreducible representations subducing into p
  for i, irrep in enumerate(irreps[:-1]):
    correlator_irrep = []
    qn_irrep = []
    # loop over gamma structure at source and sink 
    # (\gamma_i, \gamma_0\gamma_i, ...)
    for gevp_row in gamma:
      correlator_gevp_row = [] 
      qn_gevp_row = []
      for gevp_col in gamma:
        correlator_gevp_col = [] 
        qn_gevp_col = []
        # loop over rows of irrep
        for row in irrep:
          correlator_row = np.zeros((0,) + data[0].shape, dtype=np.double)
          qn_row = []
          # loop over entries in CG-coefficient: First element is the momenta, 
          # last three elements are corresponding coefficients for a spin 
          # triplet
          for el in row:
            subduced = np.zeros((1,) + data[0].shape)
            # loop over contracted operators 
            for op, qn in enumerate(qn_data):
              if np.array_equal(el[0], qn[0]):
                for g_so in range(0,3):
                  for g_si in range(0,3):
                    # Hardcoded 2pt functions here because 
                    #   sink operator = conj(source operator)
                    # and irreps are diagonal. Otherwise el[g_si] must be 
                    # calculated in seperate loop over the sink momentum
                    factor = el[g_so+1] * np.conj(el[g_si+1])
                    if (factor == 0):
                      continue

                    # calculating the Correlators yields imaginary parts in 
                    # gi-g0gi, gi-gjgk and cc. Switching real and imaginary
                    # part in the subduction factor accounts for this.
                    if (gevp_row[g_so] in gamma_i) != \
                                                    (gevp_col[g_si] in gamma_i):
                      factor = factor.imag + factor.real * 1j

                    # HARDCODED: factors I'm not certain about
#                     if(((gevp_row[g_so] in gamma_i) and \
#                                                   (gevp_col[g_si] in gamma_0i)) \
#                        or ((gevp_row[g_so] in gamma_i) and 
#                                                    (gevp_col[g_si] in gamma_0i))):
#                       factor = (-1) * factor
                  
                    if(((gevp_row[g_so] in gamma_i) and \
                                                 (gevp_col[g_si] in gamma_0i)) \
                       or ((gevp_row[g_so] in gamma_50i) and 
                                                 (gevp_col[g_si] in gamma_i))):
                      factor = 2 * factor
                    else:
                      factor = (-2) * factor

                    if (gevp_row[g_so] == qn[2]) and (gevp_col[g_si] == qn[5]):
                      subduced[0] = subduced[0] + factor.real * data[op].real \
                                                   + factor.imag * data[op].imag
                      if verbose:
                        print '\tsubduced g_so = %i, g_si = %i' % \
                                                (gevp_row[g_so], gevp_col[g_si])
                        print '\t\tsubduction coefficient = % .2f + % .2fi' % \
                                                      (factor.real, factor.imag)
            # Omit correlator if no contracted operators are contributing
            if(subduced.any() != 0):
              if verbose:
                print '\tinto momentum (%i,%i,%i)' % \
                                                  (el[0][0], el[0][1], el[0][2])
                print ' '
              correlator_row = np.vstack((correlator_row, subduced))
              qn_row.append([ el[0], (-1)*el[0], gevp_row[-1], gevp_col[-1], irreps[-1][i] ])

          correlator_gevp_col.append(correlator_row)
          qn_gevp_col.append(qn_row)
        correlator_gevp_row.append(correlator_gevp_col)
        qn_gevp_row.append(qn_gevp_col)
      correlator_irrep.append(correlator_gevp_row)
      qn_irrep.append(qn_gevp_row)
    correlator.append(np.asarray(correlator_irrep))
    qn_subduced.append(np.asarray(qn_irrep))

  correlator = np.asarray(correlator)
  print correlator.shape
  qn_subduced = np.asarray(qn_subduced)
  print qn_subduced.shape
  
  ################################################################################
  # write data to disc
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i/' % p_cm)
  utils.ensure_dir('./binarydata/')
  utils.ensure_dir('./binarydata/p%1i/' % p_cm)
  
  # write all subduced correlators
  path = './readdata/p%1i/C20_p%1i_subduced' % (p_cm, p_cm)
  np.save(path, correlator)
  path = './readdata/p%1i/C20_p%1i_subduced_qn' % (p_cm, p_cm)
  np.save(path, qn_subduced)

  # write means over all operators subducing into same irrep row
  path = './readdata/p%1i/C20_p%1i_subduced_avg_vecks' % (p_cm, p_cm)
  # TODO: mp.mean or np.sum?
  # TODO: put that into some function
#  print correlator[0,0,0].shape
  if correlator.ndim >= 5:
    avg = np.mean(correlator, axis=4)
  else:
    avg = []
    for i, irrep in enumerate(correlator):
      avg_irrep = []
      for g1, gevp_row in enumerate(irrep):
        avg_gevp_row = []
        for g2, gevp_col in enumerate(gevp_row):
          avg_gevp_col = []
          for r, row in enumerate(gevp_col):
            avg_gevp_col.append(np.sum(row, axis=0))
          avg_gevp_col = np.asarray(avg_gevp_col)
          avg_gevp_row.append(avg_gevp_col)
        avg_gevp_row = np.asarray(avg_gevp_row)
        avg_irrep.append(avg_gevp_row)
      avg_irrep = np.asarray(avg_irrep)
      avg.append(avg_irrep)
    avg = np.asarray(avg)
  print avg.shape

#  qn_avg = np.empty_like(qn_subduced[...,0,:])
  qn_avg = []
  for i, qn_irrep in enumerate(qn_subduced):
    qn_avg_irrep = []
    for g1, qn_gevp_row in enumerate(qn_irrep):
      qn_avg_gevp_row = []
      for g2, qn_gevp_col in enumerate(qn_gevp_row):
        qn_avg_gevp_col = []
        for r, qn_row in enumerate(qn_gevp_col):
          qn_avg_row = []
          for k, qn_vec in enumerate(qn_row):
#            qn_avg_row.append(np.asarray([np.dot(qn_avg_vec[1], qn_avg_vec[1]), \
#                               np.dot(qn_avg_vec[1], qn_avg_vec[1]), \
#                                                              qn_avg_vec[-3:]]))
            qn_avg_row.append(np.insert( np.insert( \
                  qn_vec[-3:], 
                    0, np.dot(qn_vec[1], qn_vec[1]), axis=-1), \
                      0, np.dot(qn_vec[0], qn_vec[0]), axis=-1))

          qn_avg_row = np.asarray(qn_avg_row)
          qn_avg_gevp_col.append(qn_avg_row)
        qn_avg_gevp_col = np.asarray(qn_avg_gevp_col)
        qn_avg_gevp_row.append(qn_avg_gevp_col)
      qn_avg_gevp_row = np.asarray(qn_avg_gevp_row)
      qn_avg_irrep.append(qn_avg_gevp_row)
    qn_avg_irrep = np.asarray(qn_avg_irrep)
    qn_avg.append(qn_avg_irrep)
  qn_avg = np.asarray(qn_avg)

#  qn_avg = np.empty_like(qn_subduced[...,0,:])
#  for i in np.ndindex(qn_subduced[...,0,0].shape):
#    qn_avg[i] = np.insert( np.insert( \
#                  qn_subduced[i][0,-3:], 
#                    np.dot(qn_subduced[i+(0,1)], qn_subduced[i+(0,1)]), 0, \
#                                                                       axis=-1), 
#                      np.dot(qn_subduced[i+(0,0)], qn_subduced[i+(0,0)]), 0, \
#                                                                        axis=-1)
  print qn_avg.shape
  np.save(path, avg)
  path = './readdata/p%1i/C20_p%1i_subduced_avg_vecks_qn' % (p_cm, p_cm)
  np.save(path, qn_avg)

   # write means over all operators subducing into same irrep
  path = './readdata/p%1i/C20_p%1i_subduced_avg_rows' % (p_cm, p_cm)
  if avg.ndim >= 4:
    avg = np.mean(avg, axis=3)
  else:
    avg_tmp = []
    for i, irrep in enumerate(avg):
      avg_irrep = []
      for g1, gevp_row in enumerate(irrep):
        avg_gevp_row = []
        for g2, gevp_col in enumerate(gevp_row):
          avg_gevp_row.append(np.mean(gevp_col, axis=0))
        avg_gevp_row = np.asarray(avg_gevp_row)
        avg_irrep.append(avg_gevp_row)
      avg_irrep = np.asarray(avg_irrep)
      avg_tmp.append(avg_irrep)
    avg = np.asarray(avg_tmp)
  print avg.shape
  np.save(path, avg)

  path = './readdata/p%1i/C20_p%1i_subduced_avg_rows_qn' % (p_cm, p_cm)
  qn_avg_tmp = []
  for i, qn_irrep in enumerate(qn_avg):
    qn_avg_irrep = []
    for g1, qn_gevp_row in enumerate(qn_irrep):
      qn_avg_gevp_row = []
      for g2, qn_gevp_col in enumerate(qn_gevp_row):
        qn_avg_gevp_row.append(qn_gevp_col[0])
      qn_avg_gevp_row = np.asarray(qn_avg_gevp_row)
      qn_avg_irrep.append(qn_avg_gevp_row)
    qn_avg_irrep = np.asarray(qn_avg_irrep)
    qn_avg_tmp.append(qn_avg_irrep)
  qn_avg = np.asarray(qn_avg_tmp)
  np.save(path, qn_avg)

#  qn_avg = qn_avg[...,0,:]
#  np.save(path, qn_avg)
 
#  # write all subduced and averaged correlators
#  # average is always over subduced operators contributing to same irrep
#  # TODO: change that to 'E2_1' in irreps[-1]
#  if p not in [1,3,4]:
#    path = './readdata/p%1i/C20_p%1i_avg_subduced' % (p, p)
#    np.save(path, np.mean(correlator, axis=1))
#    path = './readdata/p%1i/C20_p%1i_avg_subduced_quantum_numbers' % (p, p)
#    np.save(path, qn_subduced[...,0,3])
#  else:
#    # I hate the E2 irrep. 
#    # Average E2_1 and E2_2 additionally and append the result
#    path = './readdata/p%1i/C20_p%1i_avg_subduced' % (p, p)
#    E2 = np.zeros((1,) + np.mean(correlator, axis=2)[0].shape)
#    for j in range(0,correlator.shape[1]):
#      E2[0, j] = np.mean(np.vstack((correlator[1,j], correlator[2,j])), axis=0)
#    np.save(path, np.vstack((np.mean(correlator, axis=2), E2 )) )
#    path = './readdata/p%1i/C20_p%1i_avg_subduced_quantum_numbers' % (p, p)
#    E2_qn = np.zeros((1,qn_subduced.shape[1], \
#                      qn_subduced[...,-3:].shape[-1]),dtype=((str,256)))
#    E2_qn[0] = qn_subduced[0,:,0,-3:]
#    for j in range(0,correlator.shape[1]):
#      E2_qn[0,j,-1] = 'E2'
#    np.save(path, np.vstack((qn_subduced[...,0,-3:], E2_qn)) )
#  
#  ################################################################################
#  # write all subduced correlators for each irrep and gamma seperately in ascii 
#  # format
#  if(correlator.shape[0] != (len(irreps)-1) ):
#    print '\tThe number of calculated irreps is not the number of existing ' \
#          'ones!'
#    exit(0)
#  if(correlator.shape[1] != len(gamma)*len(gamma) ):
#    print '\tThe number of calculated gamma combinations is not the number of ' \
#          'existing ones!'
#    exit(0)
  
#  #TODO: finish output to binary files with dictionary
#  #TODO: somehow make this dictionary more elegant
#  if p not in [1,3,4]:
#    nb_irreps = correlator.shape[0]
#  else:
#    # Average E2_1 and E2_2 additionally and append the result
#    for j in ( range(0, correlator.shape[1]) ):
#      path = './binarydata/p%1i/C20_p%1i_E2_%s_%s.dat' % (p, p, \
#                        gamma_for_filenames[qn_subduced[i,j,0,-3]], \
#                        gamma_for_filenames[qn_subduced[i,j,0,-2]])
#      f = open(path, 'wb')
#      E2 = np.mean(np.vstack((correlator[-1,j], correlator[-2,j])), axis=0)
#      (E2.swapaxes(-1, -2)).tofile(f)
#      f.close()
#    nb_irreps = correlator.shape[0]-2
#
#  for i in ( range(0, nb_irreps) ):
#    for j in ( range(0, correlator.shape[1]) ):
#      # write in format p_irrep_gso_gsi
#      path = './binarydata/p%1i/C20_p%1i_%s_%s_%s.dat' % (p, p, \
#                        qn_subduced[i,j,0,-1], \
#                        gamma_for_filenames[qn_subduced[i,j,0,-3]], \
#                        gamma_for_filenames[qn_subduced[i,j,0,-2]])
#      f = open(path, 'wb')
#      (correlator[i,j].swapaxes(-1, -2)).tofile(f)
#      f.close()
 
  ################################################################################
  # write all subduced correlators for each irrep and gamma seperately
  #if(correlator.shape[0] != (len(irreps)-1) ):
  #  print '\tThe number of calculated irreps is not the number of existing ' \
  #        'ones!'
  #  exit(0)
  #if(correlator.shape[1] != len(gamma)*len(gamma) ):
  #  print '\tThe number of calculated gamma combinations is not the number of ' \
  #        'existing ones!'
  #  exit(0)
  #
  #for i in ( range(0, correlator.shape[0]) ):
  #  for j in ( range(0, correlator.shape[1]) ):
  #   path = './readdata/p%1i/single/C20_p%1i_single_%s_%s_%s' % (p, p, \
  #           qn_subduced[i,j,0,-3], qn_subduced[i,j,0,-2], \
  #                                               qn_subduced[i,j,0,-1])
  #   np.save(path, correlator[i])
  #   path = './readdata/p%1i/single/C20_p%1i_single_%s_%s_%squantum_numbers' % (p, p, \
  #           qn_subduced[i,j,0,-3], qn_subduced[i,j,0,-2], \
  #                                               qn_subduced[i,j,0,-1])
  #   np.save(path, qn_subduced[i])

for p_cm in p:
  subduce_ensembles(p_cm, gamma, verbose=0)


