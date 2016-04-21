#!/usr/bin/python

import os, glob
import struct

import numpy as np

import sys
#sys.stdout = open('./read.out', 'w')

import utils

################################################################################
# parameters ###################################################################

sta_cnfg = 714
end_cnfg = 1214
del_cnfg = 2

T = 48        # number of timeslices
p = 0         # momentum

verbose = 0

#directory = '/hiskp2/werner/analyse_jan/A40/data/'
#directory = '/hiskp2/knippsch/Test_rho_2pt/'
directory = '/hiskp2/knippsch/Rho_Feb2016/'

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

################################################################################
# reading configurations

def read_ensembles(sta_cnfg, end_cnfg, del_cnfg, p, T, directory, missing_configs, verbose=0):
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
    if i % (nb_cnfg/10) == 0:
      print '\tread config %i' % i
    data_cnfg = np.zeros((0, T), dtype=np.complex)
    qn_cnfg = []
    # only walk through directories containing the configuration number or a 'C'
    # (for Correlator)
    for root, dirs, files in os.walk(directory, topdown=True):
      c = ['cnfg%i' % i, 'C20', 'first']
      dirs[:] = [d for d in dirs for cc in c if d.find(cc) !=-1]
      # sorted ensures same order of operators for every cnfg
      for name in sorted(files):
        # running over all subdirectories and searching for files starting with 
        # "C20_uu"
        if name.startswith('C20_uu'):
        # filename and path
          filename = os.path.join(root, name)

          # getting momentum, displacement and gamma structure from name
          split = name.replace('_p', ' ').replace('.d', ' '). \
                  replace('.g', ' ').replace('.dat', ' ').split()
          ensemble_data = [split_to_vector(split[1]), split_to_vector(split[2]), \
                           np.asarray(split[3], dtype=int), \
                           split_to_vector(split[4]), split_to_vector(split[5]), \
                           np.asarray(split[6], dtype=int) ]
  
          # ensure p is correct
          if not ( (np.dot(ensemble_data[0], ensemble_data[0]) == p) and \
                   (np.dot(ensemble_data[3], ensemble_data[3]) == p) ):
            continue
  
          read_data = np.zeros(T, dtype=np.complex)

          if verbose:
            print 'Reading data from file:'
            print '\t\t' + filename
          try:
            f = open(filename, 'rb')
          except IOError:
            continue
         
          # actual reading of complex number
          for t in range(0,T):
            read_data[t] = complex(struct.unpack('d', f.read(8))[0], \
                                   struct.unpack('d', f.read(8))[0])
          f.close()
          data_cnfg = np.vstack((data_cnfg, read_data))
          # set up quantum numbers and reference shape in first iteration
          qn_cnfg.append(ensemble_data)
  
    # check if number of operators is consistent between configurations and 
    # operators are identical
    qn_cnfg = np.asarray(qn_cnfg)
    if(cnfg == 0):
      shape = data_cnfg.shape
      quantum_numbers = qn_cnfg
    else:
      if(data_cnfg.shape != shape):
        print 'Wrong number of operators for cnfg %i' % i
        exit(0)
      if(quantum_numbers.shape != qn_cnfg.shape):
        print 'Wrong operator for cnfg %i' %i
        exit(0)
      for q,r in zip(quantum_numbers.flatten(), qn_cnfg.flatten()):
        if not np.array_equal(q, r):
          print 'Wrong operator for cnfg %i' %i
          exit(0)
  
    data[cnfg] = data_cnfg
    cnfg = cnfg + 1
  
  # convert data to a 3-dim np-array with nb_op x x T x nb_cnfg 
  data = np.asarray(data)
  data = np.rollaxis(data, 0, 3)
  print data.shape
  
  print '\tfinished reading'
  
  ################################################################################
  # write data to disc
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i' % p)
  utils.ensure_dir('./readdata/p%1i/single' % p)
  # write every operator seperately
  for i in range(0, data.shape[0]):
    path = './readdata/p%1i/single/C20_p%i%i%i.d%i%i%i.g%i_p%i%i%i.d%i%i%i.g%i' % \
            (p, quantum_numbers[i][0][0], quantum_numbers[i][0][1], 
             quantum_numbers[i][0][2], \
             quantum_numbers[i][1][0], quantum_numbers[i][1][1], 
             quantum_numbers[i][1][2], quantum_numbers[i][2],
             quantum_numbers[i][3][0], quantum_numbers[i][3][1], 
             quantum_numbers[i][3][2], \
             quantum_numbers[i][4][0], quantum_numbers[i][4][1], 
             quantum_numbers[i][4][2], quantum_numbers[i][5])
    np.save(path, data[i])
  
  # write all operators
  path = './readdata/p%1i/C20_p%1i' % (p, p)
  np.save(path, data)
  
  # write all quantum numbers
  path = './readdata/p%1i/C20_p%1i_quantum_numbers' % (p, p)
  np.save(path, quantum_numbers)
  
  print '\tfinished writing'
  
  ################################################################################
  # Code to average operators before bootstrapping. Necessary, if correlators 
  # are multiplied.
  
  #print data.shape
  #shape = data[0].shape
  #data = np.reshape(data, (data.shape[0], ) +  data[0].flatten().shape)
  #
  #gamma_i = [1, 2, 3]
  #gamma_0i = [10, 11, 12]
  #gamma = [gamma_i, gamma_0i, ['g_i', 'g_0g_i']]
  #
  #avg = [[None]*(len(gamma)-1)]*(len(gamma)-1)
  #for i in range(0, len(gamma)-1):
  #  for j in range(0, len(gamma)-1):
  #    same_operators = np.zeros((0, data.shape[1]), dtype=np.complex)
  #    for op in range(0, data.shape[0]):
  #      if ( (quantum_numbers[op][2] in gamma[i]) and (quantum_numbers[op][5] in gamma[j]) ):
  #        print same_operators.shape
  #        print data[op].shape
  #        same_operators = np.vstack((same_operators, data[op]))
  #        if(op == 0):
  #    same_operators = np.reshape(same_operators, (same_operators.shape[0], ) + shape)
  #    avg[i][j] = np.mean(same_operators, axis=0)
  #data = np.reshape(data, (data.shape[0], ) + shape)
  #
  #avg = np.asarray(avg)
  #print avg.shape
  
  # bootstrap procedure for real and imaginary part seperately
  # if used, put that into bootstrap.py
  #avg_real = bootstrap(avg.real, nb_boot)
  #avg_imag = bootstrap(avg.imag, nb_boot)
  #print avg.shape

#read_ensembles(sta_cnfg, end_cnfg, del_cnfg, p, T, directory, missing_configs, verbose=0)
