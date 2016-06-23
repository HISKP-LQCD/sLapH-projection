#!/usr/bin/python

import os, glob
import struct

import h5py
import numpy as np
import itertools as it
import operator

import sys
#sys.stdout = open('./read.out', 'w')

import utils

################################################################################
# parameters ###################################################################

sta_cnfg = 714
end_cnfg = 1214
del_cnfg = 2

T = 48        # number of timeslices
p_max = 4
p = range(1)         # momentum

# gamma structure wich shall be averaged. Last entry of gamma must contain all
# names in LaTeX compatible notation for plot labels
gamma_i =   [1, 2, 3, \
             ['\gamma_1', '\gamma_2', '\gamma_3', '\gamma_i']]
gamma_0i =  [10, 11, 12, \
             ['\gamma_0\gamma_1', '\gamma_0\gamma_2', '\gamma_0\gamma_3', \
              '\gamma_0\gamma_i']]
gamma_50i = [13, 14, 15, \
             ['\gamma_5\gamma_0\gamma_1', '\gamma_5\gamma_0\gamma_2', \
              '\gamma_5\gamma_0\gamma_3', '\gamma_5\gamma_0\gamma_i']]
gamma_5 = [5, ['\gamma_5']]

gammas = [gamma_i, gamma_0i, gamma_50i]

diagrams = ['C20']

verbose = 0

#directory = '/hiskp2/werner/analyse_jan/A40/data/'
#directory = '/hiskp2/knippsch/Test_rho_2pt/'
directory = '/hiskp2/knippsch/Rho_Jun2016/'

missing_configs = [1282]

###############################################################################
# short function that splits a string of three signed integers into a list
def split_to_vector(string):
  vector = np.zeros(3, dtype=np.int)
  i = 0
  # loop over components
  for j in range(0, 3):
    # append negative number (slicing because of sign)
    if(not string[i].isdigit()):
      vector[j] = int(string[i:i+2])
      i = i+1
    # if not negative, append positive number
    else:
      vector[j] = int(string[i])
    i = i+1
  return vector

def scalar_mul(x, y):
  return sum(it.imap(operator.mul, x, y))

def abs2(x):
  return scalar_mul(x, x)

def set_lookup_p(p_cm, diagram):

  # create lookup table for all possible 3-momenta that can appear in our 
  # contractions
  lookup_p3 = it.ifilter(lambda x: abs2(x) <= p_max, \
                                 it.product(range(-p_max, p_max+1), repeat=3))
  
  lookup_p3, tmp = it.tee(lookup_p3, 2)
  tmp = list(it.ifilter(lambda x : abs2(x) == p_cm, tmp))
  if diagram == 'C20':
    lookup_p = it.ifilter(lambda (x,y): \
               # if momentum is the desired center of mass momentum and 
               # respects momentum conservation
               (tuple(x) == tuple(it.imap(operator.neg, y))) and \
                                                            (tuple(x) in tmp), \
               # create lookup table with all possible combinations 
               # of three 3-momenta 
                                                 it.product(lookup_p3, repeat=2))
  else:
    print 'in set_lookup_p: diagram unknown! Quantum numbers corrupted.'
  
  return list(lookup_p)

def set_qn(p, g, diagram):

  #contains list with momentum, displacement, gamma for both, source and sink
  if diagram == 'C20':
    return [p[0], np.zeros((3,)), np.asarray(g[0], dtype=int), \
            p[1], np.zeros((3,)), np.asarray(g[1], dtype=int)]
  else:
    print 'in set_qn: diagram unknown! Quantum numbers corrupted.'
  return

def set_lookup_g():
  # all combinations, but only g1 with g01 etc. is wanted
  lookup_g = it.product([g for gamma in gammas for g in gamma[:-1]], repeat=2)
#  indices = [[1,2,3],[10,11,12],[13,14,15]]
#  lookup_g2 = [list(it.product([i[j] for i in indices], repeat=2)) for j in range(len(indices[0]))]
#  lookup_g = [item for sublist in lookup_g2 for item in sublist]

  return list(lookup_g)

################################################################################
# reading configurations

def read_ensembles(sta_cnfg, end_cnfg, del_cnfg, diagram, p, T, directory, \
                                                    missing_configs, verbose=0):
  
  print 'reading data for %s, p=%i' % (diagram, p_cm)

  # calculate number of configurations
  nb_cnfg = 0
  for i in range(sta_cnfg, end_cnfg+1, del_cnfg):
    if i in missing_configs:
      continue
    nb_cnfg = nb_cnfg + 1
  if(verbose):
    print 'number of configurations: %i' % nb_cnfg

  lookup_p = set_lookup_p(p_cm, diagram)
  lookup_g = set_lookup_g()
  
  data = []
  qn = []

  cnfg_counter = 0
  for cnfg in range(sta_cnfg, end_cnfg+1, del_cnfg):
    if cnfg in missing_configs:
      continue
#    if i % (nb_cnfg/10) == 0:
#      print '\tread config %i' % i
 
    # filename and path
    filename = directory + 'cnfg%i/' % cnfg + diagram + '_cnfg%i' % cnfg + '.h5'
#    name = 
#    filename = os.path.join(path, name)
    if verbose:
      print filename

    f = h5py.File(filename, 'r')

    data_cnfg = []
    for p in lookup_p:
      for g in lookup_g:

        groupname = diagram + '_uu_p%1i%1i%1i.d000.g%i' % \
                                             (p[0][0], p[0][1], p[0][2], g[0]) \
                    + '_p%1i%1i%1i.d000.g%i' % (p[1][0], p[1][1], p[1][2], g[1])

        if verbose:
          print groupname

        data_cnfg.append(np.asarray(f[groupname]).view(np.complex))

        if cnfg_counter == 0:
          qn.append(set_qn(p, g, diagram))
    data_cnfg = np.asarray(data_cnfg)
    data.append(data_cnfg)

    f.close()

    cnfg_counter = cnfg_counter + 1

  data = np.asarray(data)
  qn = np.asarray(qn)
  
#    # check if number of operators is consistent between configurations and 
#    # operators are identical
#    data_cnfg = np.asarray(data_cnfg)
#    qn_cnfg = np.asarray(qn_cnfg)
#    if(cnfg == 0):
#      shape = data_cnfg.shape
#      quantum_numbers = qn_cnfg
#    else:
#      if(data_cnfg.shape != shape):
#        print 'Wrong number of operators for cnfg %i' % i
#        exit(0)
#      if(quantum_numbers.shape != qn_cnfg.shape):
#        print 'Wrong operator for cnfg %i' %i
#        exit(0)
#      for q,r in zip(quantum_numbers.flatten(), qn_cnfg.flatten()):
#        if not np.array_equal(q, r):
#          print 'Wrong operator for cnfg %i' %i
#          exit(0)
#  
#    data[cnfg] = data_cnfg
#    cnfg = cnfg + 1
#  
  # convert data to a 3-dim np-array with nb_op x x T x nb_cnfg 
  data = np.asarray(data)
  data = np.rollaxis(data, 0, 3)
  print data.shape
  
  print '\tfinished reading'
  
  ################################################################################
  # write data to disc
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i' % p_cm)

  # write all operators
  path = './readdata/p%1i/%s_p%1i' % (p_cm, diagram, p_cm)
  np.save(path, data)
  
  # write all quantum numbers
  path = './readdata/p%1i/%s_p%1i_qn' % (p_cm, diagram, p_cm)
  np.save(path, qn)
  
  print '\tfinished writing'

  
#  # write every operator seperately
#  utils.ensure_dir('./readdata/p%1i/single' % p)
#  for i in range(0, data.shape[0]):
#    path = './readdata/p%1i/single/%s' % \
#            (p, quantum_numbers[i][-1])
#    np.save(path, data[i])



#
# 
#  
#  ################################################################################
#  # Code to average operators before bootstrapping. Necessary, if correlators 
#  # are multiplied.
#  
#  #print data.shape
#  #shape = data[0].shape
#  #data = np.reshape(data, (data.shape[0], ) +  data[0].flatten().shape)
#  #
#  #gamma_i = [1, 2, 3]
#  #gamma_0i = [10, 11, 12]
#  #gamma = [gamma_i, gamma_0i, ['g_i', 'g_0g_i']]
#  #
#  #avg = [[None]*(len(gamma)-1)]*(len(gamma)-1)
#  #for i in range(0, len(gamma)-1):
#  #  for j in range(0, len(gamma)-1):
#  #    same_operators = np.zeros((0, data.shape[1]), dtype=np.complex)
#  #    for op in range(0, data.shape[0]):
#  #      if ( (quantum_numbers[op][2] in gamma[i]) and (quantum_numbers[op][5] in gamma[j]) ):
#  #        print same_operators.shape
#  #        print data[op].shape
#  #        same_operators = np.vstack((same_operators, data[op]))
#  #        if(op == 0):
#  #    same_operators = np.reshape(same_operators, (same_operators.shape[0], ) + shape)
#  #    avg[i][j] = np.mean(same_operators, axis=0)
#  #data = np.reshape(data, (data.shape[0], ) + shape)
#  #
#  #avg = np.asarray(avg)
#  #print avg.shape
#  
#  # bootstrap procedure for real and imaginary part seperately
#  # if used, put that into bootstrap.py
#  #avg_real = bootstrap(avg.real, nb_boot)
#  #avg_imag = bootstrap(avg.imag, nb_boot)
#  #print avg.shape

for p_cm in p:
  for diagram in diagrams:
    read_ensembles(sta_cnfg, end_cnfg, del_cnfg, diagram, p_cm, T, directory, \
        missing_configs, verbose)
