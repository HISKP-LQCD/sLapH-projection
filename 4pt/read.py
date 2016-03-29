#!/usr/bin/python

#itertools
import os
import struct

import numpy as np

import utils

################################################################################
# parameters ###################################################################

sta_cnfg = 714
end_cnfg = 1214
del_cnfg = 2

T = 48
p_cm = 0
p_max = 4

p_cm_max = np.asarray((6,5,6,7,4), dtype=int)
lookup_p3_reduced = np.asarray((np.asarray((0,0,0), dtype=int), \
                           np.asarray((0,0,1), dtype=int), \
                           np.asarray((0,1,1), dtype=int), \
                           np.asarray((1,1,1), dtype=int), \
                           np.asarray((0,0,2), dtype=int) ))

directory = '/hiskp2/knippsch/Rho_Feb2016/'

missing_configs = [1282]

verbose = 0

################################################################################

#TODO: create np-array with all necessary momenta in z-direction -> dudek paper

count = 0
for p_cm in range(0, 5):
  print 'p_cm = %i' % p_cm
  # create lookup table for all possible 3-momenta that can appear in our 
  # contractions
  lookup_p3 = np.zeros((0, 3), dtype=int)
  for px in range(-p_max, p_max+1):
    for py in range(-p_max, p_max+1):
      for pz in range(-p_max, p_max+1):
        p3 = np.asarray((px, py, pz), dtype = int)
        if np.dot(p3, p3) <= p_max:
          lookup_p3 = np.vstack((lookup_p3, p3))
 
  # create lookup table for all possible sums of 3-momenta that give the right 
  # center-of-mass momentum 
  lookup_p = []
  lookup_p_reduced = []
  for p1 in lookup_p3:
    for p2 in lookup_p3:
      if ((np.dot(p1,p1) + np.dot(p2,p2)) > p_cm_max[p_cm]):
        continue
      if (np.dot( (p1+p2), (p1+p2)) == p_cm):
        lookup_p.append(np.vstack((p1, p2)) )
      if (np.array_equal((p1+p2), lookup_p3_reduced[p_cm])): 
        if p_cm == 0 and np.array_equal(p1, p2):
          continue
        lookup_p_reduced.append(np.vstack((p1, p2)) )

  lookup_p = np.asarray(lookup_p)
  lookup_p_reduced = np.asarray(lookup_p_reduced)

  if verbose:
    print lookup_p.shape
    print lookup_p_reduced.shape


  # create lookup table with all possible 3-momentum combinations that generate
  # 4pt functions with the correct center-of-mass momentum and respect momentum
  # conservation

  # direct diagram
  lookup_p_d = []
  for p_so in lookup_p:
    for p_si in lookup_p:
      if (np.array_equal(p_so[0]+p_so[1], p_si[0]+p_si[1])):
        lookup_p_d.append(np.vstack((p_so[0], -p_si[0], p_so[1], -p_si[1])) )

  # box diagram
  lookup_p_b = []
  for p_so in lookup_p_reduced:
    for p_si in lookup_p_reduced:
#      if p_cm != 0 or (np.dot(p_so[0],p_so[0]) ==  np.dot(p_si[0],p_si[0])):
      if ( (p_cm != 0) or (np.dot(p_so[0], p_so[0]) < 3 and np.dot(p_si[0], p_si[0]) < 3) ):
        lookup_p_b.append(np.vstack((p_so[0], -p_si[0], -p_si[1], p_so[1])) )

  lookup_p_b = np.asarray(lookup_p_b)
  lookup_p_d = np.asarray(lookup_p_d)
  
  if verbose:
    print lookup_p_d.shape
    print lookup_p_b.shape
    print ' '
  
#for p_test in lookup_p3:
#  counter = 0
#  for i in range(lookup_p_b.shape[0]):
#    if np.array_equal(lookup_p_b[i, 0], p_test):
##      print lookup_p_b[i]
##      print ' '
#      counter = counter + 1
#  print p_test
#  print counter

#for i in range(lookup_p_b.shape[0]):
#  if np.array_equal(lookup_p_b[i, 0], np.asarray((0,-1,0))):
##    if np.array_equal(lookup_p_b[i, 1], np.asarray((0,-1,0))):
#    print lookup_p_b[i]
#    print ' '

  # calculate number of configurations
  nb_cnfg = 0
  for i in range(sta_cnfg, end_cnfg+1, del_cnfg):
    if i in missing_configs:
      continue
    nb_cnfg = nb_cnfg + 1
  if(verbose):
    print 'number of configurations: %i' % nb_cnfg

  #TODO: check if all files are there
  data = [None]*nb_cnfg
  cnfg = 0

  for i in range(sta_cnfg, end_cnfg+1, del_cnfg):
    if i in missing_configs:
      continue
    if i % max(1,(nb_cnfg/10)) == 0:
      print '\tread config %i' % i
    data_cnfg = np.zeros((0, T), dtype=np.complex)
    qn_cnfg = []
    for p in lookup_p_b:
      diagram = 'C4+B'
      path = directory + 'cnfg%i/' % i + 'cnfg%i/' % i + diagram + \
             '/first_p_%1i%1i%1i/' % (p[0, 0], p[0, 1], p[0, 2])
      name = diagram + '_uuuu_p%1i%1i%1i.d000.g5' % (p[0, 0], p[0, 1], p[0, 2]) + \
             '_p%1i%1i%1i.d000.g5' % (p[1, 0], p[1, 1], p[1, 2]) + \
             '_p%1i%1i%1i.d000.g5' % (p[2, 0], p[2, 1], p[2, 2]) + \
             '_p%1i%1i%1i.d000.g5' % (p[3, 0], p[3, 1], p[3, 2]) + '.dat'
    
      # filename and path
      filename = os.path.join(path, name)
      if verbose:
        print 'Reading data from file:'
        print '\t\t' + filename

      if cnfg == 0:
        # set up quantum numbers and reference shape in first iteration
        ensemble_data = [p[0], np.zeros((3,)), np.asarray(5, dtype=int), \
                         p[1], np.zeros((3,)), np.asarray(5, dtype=int), \
                         p[2], np.zeros((3,)), np.asarray(5, dtype=int), \
                         p[3], np.zeros((3,)), np.asarray(5, dtype=int), name]
        qn_cnfg.append(ensemble_data)
        quantum_numbers = np.asarray(qn_cnfg)
  
      read_data = np.zeros(T, dtype=np.complex)
  
      try:
        f = open(filename, 'rb')
      except IOError:
        print '\tFailed to open %s' % filename
        continue
      if verbose:
        print f
      
      # actual reading of complex number
      for t in range(0,T):
        read_data[t] = complex(struct.unpack('d', f.read(8))[0], \
                               struct.unpack('d', f.read(8))[0])
      f.close()
      data_cnfg = np.vstack((data_cnfg, read_data))
  
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
  
    data[cnfg] = data_cnfg
    cnfg = cnfg + 1
  
  # convert data to a 3-dim np-array with nb_op x x T x nb_cnfg 
  data = np.asarray(data)
  data = np.rollaxis(data, 0, 3)
  print data.shape
  print quantum_numbers.shape

  print '\tfinished reading'
  
  ################################################################################
  # write data to disc
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i' % p_cm)
  utils.ensure_dir('./readdata/p%1i/single' % p_cm)
  # write every operator seperately
  for i in range(0, quantum_numbers.shape[0]):
    path = './readdata/p%1i/single/%s' % \
            (p_cm, quantum_numbers[i][-1])
#             quantum_numbers[i][0][0], quantum_numbers[i][0][1], 
#             quantum_numbers[i][0][2], \
#             quantum_numbers[i][1][0], quantum_numbers[i][1][1], 
#             quantum_numbers[i][1][2], quantum_numbers[i][2],
#             quantum_numbers[i][3][0], quantum_numbers[i][3][1], 
#             quantum_numbers[i][3][2], \
#             quantum_numbers[i][4][0], quantum_numbers[i][4][1], 
#             quantum_numbers[i][4][2], quantum_numbers[i][5])
    np.save(path, data[i])
  
  # write all operators
  path = './readdata/p%1i/C20_p%1i' % (p_cm, p_cm)
  np.save(path, data)
  
  # write all quantum numbers
  path = './readdata/p%1i/C20_p%1i_quantum_numbers' % (p_cm, p_cm)
  np.save(path, quantum_numbers)
  
  print '\tfinished writing'

