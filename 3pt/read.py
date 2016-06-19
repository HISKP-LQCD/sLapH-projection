#!/usr/bin/python

import os
import struct

import numpy as np
import itertools as it
import operator

import utils

################################################################################
# parameters ###################################################################

sta_cnfg = 714
end_cnfg = 1214
del_cnfg = 2

T = 48
p = range(5)
p_max = 4

p_cm_max = np.asarray((4,5,6,7,4), dtype=int)
lookup_p3_reduced = [(0,0,0), (0,0,1), (0,1,1), (1,1,1), (0,0,2)]

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

diagrams = ['C3+']

directory = ['/hiskp2/werner/Rho_May2016/']

missing_configs = [1282]

verbose = 0

################################################################################

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
  if diagram == 'C3+':
    lookup_p = it.ifilter(lambda (w,x,y): \
               # if the sum of both momenta is the desired center of mass 
               # momentum and momentum conservation
#               tuple(it.imap(operator.add, w, y)) == lookup_p3_reduced[p_cm] \
#               and tuple(it.imap(operator.neg, x)) \
#                                                  == lookup_p3_reduced[p_cm] \
#
               (tuple(it.imap(operator.add, w, y)) == tuple(it.imap(operator.neg, x))) \
               and tuple(it.imap(operator.add, w, y)) in tmp \

               # for zero center of mass momentum omit the case were both 
               # particles are at rest (s-wave)
               and not (p_cm == 0 and (tuple(w) == tuple(y))) \
               # omit cases for which both particles carry high momentum 
               # cancelling to the desired cm-momentum
               and (abs2(w) + abs2(y) <= p_cm_max[p_cm]), \
                        # create lookup table with all possible combinations 
                        # of three 3-momenta 
                        it.product(lookup_p3, repeat=3))
  else:
    print 'in set_lookup_p: diagram unknown! Quantum numbers corrupted.'

#    #direct diagram
#    lookup_p_d = []
#    for p_so in lookup_p:
#      for p_si in lookup_p:
#        if (np.array_equal(p_so[0]+p_so[1], p_si[0]+p_si[1])):
#          lookup_p_d.append(np.vstack((p_so[0], -p_si[0], p_so[1], -p_si[1])) )

  return lookup_p
  
def set_qn(p, g, name, diagram):

  # switch momenta so that the first two momenta are always at the source and
  # the last two at sink, independent of the quark line flow
  if diagram == 'C3+':
    return [p[0], np.zeros((3,)), np.asarray(5, dtype=int), \
            p[1], np.zeros((3,)), np.asarray(g, dtype=int), \
            p[2], np.zeros((3,)), np.asarray(5, dtype=int), name]
  else:
    print 'in set_qn: diagram unknown! Quantum numbers corrupted.'
  return

#TODO: create np-array with all necessary momenta in z-direction -> dudek paper

for p_cm in p:
  print 'p_cm = %i' % p_cm
  for d, diagram in enumerate(diagrams):
    print '\tread diagram %s' % diagram

    # create lookup table with all possible 3-momentum combinations that 
    # generate 4pt functions with the correct center-of-mass momentum and 
    # respect momentum conservation
    lookup_p = set_lookup_p(p_cm, diagram)
  
    # calculate number of configurations
    nb_cnfg = 0
    for i in range(sta_cnfg, end_cnfg+1, del_cnfg):
      if i in missing_configs:
        continue
      nb_cnfg = nb_cnfg + 1
    if(verbose):
      print 'number of configurations: %i' % nb_cnfg
  
    #TODO: check if all files are there
#    data = [None]*nb_cnfg
    data = []
    ensemble_data = []
    cnfg = 0
    for i in range(sta_cnfg, end_cnfg+1, del_cnfg):
      if i in missing_configs:
        continue
      if i % max(1,(nb_cnfg/10)) == 0:
        print '\tread config %i' % i

#      data_cnfg = np.zeros((0, T), dtype=np.complex)
      data_cnfg = []

      for gamma in gammas:
        for g in gamma[:-1]:
          itera, lookup_p = it.tee(lookup_p, 2)
          for p in itera:
  
            # filename and path
            path = directory[d] + 'cnfg%i/' % i + 'cnfg%i/' % i + diagram + \
                   '/first_p_%1i%1i%1i/' % (p[0][0], p[0][1], p[0][2])
            name = diagram + '_uuu_p%1i%1i%1i.d000.g5' % (p[0][0], p[0][1], \
                                                                        p[0][2]) + \
                   '_p%1i%1i%1i.d000.g%1i' % (p[1][0], p[1][1], p[1][2], g) + \
                   '_p%1i%1i%1i.d000.g5' % (p[2][0], p[2][1], p[2][2]) + '.dat'
            filename = os.path.join(path, name)
            if verbose:
              print 'Reading data from file:'
              print '\t\t' + filename
  
            # open file
            try:
              f = open(filename, 'rb')
            except IOError:
              print '\tFailed to open %s' % filename
              continue
            if verbose:
              print f
   
            # set up quantum numbers and reference shape in first iteration
            if cnfg == 0:
              ensemble_data.append(set_qn(p, g, name, diagram))
            
            # actual reading of complex number
            read_data = np.zeros(T, dtype=np.complex)
            for t in range(0,T):
              read_data[t] = complex(struct.unpack('d', f.read(8))[0], \
                                     struct.unpack('d', f.read(8))[0])
            f.close()
#            data_cnfg = np.vstack((data_cnfg, read_data))
            data_cnfg.append(read_data)
      data_cnfg = np.asarray(data_cnfg) 
      # check if number of operators is consistent between configurations and 
      # operators are identical
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
    
      data.append(data_cnfg)
      cnfg = cnfg + 1
    
    quantum_numbers = np.asarray(ensemble_data)
    # convert data to a 3-dim np-array with nb_op x T x nb_cnfg 
    data = np.asarray(data)
    data = np.rollaxis(data, 0, 3)

    print quantum_numbers.shape
    print data.shape
  
    print '\tfinished reading'
    
    ################################################################################
    # write data to disc
    
    utils.ensure_dir('./readdata')
    utils.ensure_dir('./readdata/p%1i' % p_cm)
    utils.ensure_dir('./readdata/p%1i/single' % p_cm)
    utils.ensure_dir('./readdata/p%1i/single/%s' % (p_cm, diagram))
    # write every operator seperately
    for i in range(0, quantum_numbers.shape[0]):
      path = './readdata/p%1i/single/%s/%s' % \
              (p_cm, diagram, quantum_numbers[i][-1])
      np.save(path, data[i])
    
    # write all operators
    path = './readdata/p%1i/%s_p%1i' % (p_cm, diagram, p_cm)
    np.save(path, data)
    
    # write all quantum numbers
    path = './readdata/p%1i/%s_p%1i_quantum_numbers' % (p_cm, diagram, p_cm)
    np.save(path, quantum_numbers)
    
    print '\tfinished writing\n'
  
