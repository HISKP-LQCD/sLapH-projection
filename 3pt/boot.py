#!/usr/bin/python

import numpy as np

import sys

import utils
#sys.stdout = open('./boot.out', 'w')

################################################################################
# Parameters ###################################################################

################################################################################
bootstrap_original_data = False

p = 0         # momentum

nb_bins = 1
nb_boot = 500

verbose = 0

################################################################################
# Functions ####################################################################

################################################################################
# prebinning
def prebinning(X, binsize):
  if verbose:
    print X.shape
  nb_bins = int(X.shape[-1]/binsize)
  if X.shape[-1] % binsize > 0:
    nb_bins = nb_bins+1
  if verbose:
    print nb_bins
  bins = np.zeros(X[...,0].shape +  (nb_bins,))
  for i in range(0, nb_bins-1):
    bins[...,i] = np.mean(X[...,i*binsize:(i+1)*binsize], axis=-1)
  bins[...,nb_bins-1] = np.mean(X[...,(nb_bins-1)*binsize:], axis=-1)
  return bins

################################################################################
# bootstrap an arbitrary array with nb_cnfg as last dimension
def bootstrap(X, boot_size, verbose=0):
  np.random.seed(1227)
  boot = np.zeros(X[...,0].shape + (boot_size, ), dtype=float)
  length = X.shape[-1]
  # writing the mean value in the first sample
  boot[...,0] = np.mean(X, axis=-1) 
  # doing all other samples
  for i in range(1, boot_size): 
    if verbose and (i % (boot_size/10) == 0):
      print '\tbootstrap sample %i' % i
    rnd = np.random.random_integers(0, high=length-1, size=length)
    boot[...,i] = np.mean(X[...,rnd], axis=-1)
  return boot

################################################################################
# Write data to disc ###########################################################
 
def write_ensemble(boot, qn_boot, name, p, write_mean=False):
 
  ################################################################################
  utils.ensure_dir('./bootdata')
  utils.ensure_dir('./bootdata/p%1i' % p)
  
 
  # write all operators
  path = './bootdata/p%1i/%s' % (p, name)
  np.save(path, boot)
  path = './bootdata/p%1i/%s_qn' % (p, name)
  np.save(path, qn_boot)
 
  # write all subduced correlators
  if write_mean:
    utils.ensure_dir('./bootdata/p%1i/avg' % p)

    print '\taveraging over momenta'
    # write means over all three-vectors of operators subducing into same irrep, 
    # [k1,k2]-Gamma, mu
    path = './bootdata/p%1i/%s_avg_vecks' % (p, name)
    if boot.ndim >= 5:
      avg = np.mean(boot, axis=4)
    else:
      avg = []
      for i, irrep in enumerate(boot):
        avg_irrep = []
        for k1k2, gevp_row in enumerate(irrep):
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
    np.save(path, avg)

#    avg = np.zeros_like(boot)
#    qn_avg = np.zeros_like(qn_boot)
#    for i in range(boot.shape[0]):
#      for k in range(boot.shape[1]):
#        for g in range(boot.shape[2]):
#          for r in range(boot.shape[3]):
#            avg[i,k,g,r] = np.sum(boot[i,k,g,r], axis=0) 
#            qn_avg[i,k,g,r] = qn_boot[i,k,g,r][0,3:]
#    avg = np.asarray(avg.tolist())
#    qn_avg = np.asarray(qn_avg.tolist())

    path = './bootdata/p%1i/%s_avg_vecks_qn' % (p, name)
    qn_avg = []
    for i, qn_irrep in enumerate(qn_boot):
      qn_avg_irrep = []
      for g1, qn_gevp_row in enumerate(qn_irrep):
        qn_avg_gevp_row = []
        for g2, qn_gevp_col in enumerate(qn_gevp_row):
          qn_avg_gevp_col = []
          for r, qn_row in enumerate(qn_gevp_col):
#            qn_avg_row = []
#            for k, qn_vec in enumerate(qn_row):
  #            qn_avg_row.append(np.asarray([np.dot(qn_avg_vec[1], qn_avg_vec[1]), \
  #                               np.dot(qn_avg_vec[1], qn_avg_vec[1]), \
  #                                                              qn_avg_vec[-3:]]))
#              qn_avg_row.append(np.insert( \
#                  np.insert( qn_vec[-5:], 0, \
#                      (np.dot(qn_vec[-7][1], qn_vec[-7][1]), np.dot(qn_vec[-7][0], qn_vec[-7][0])), \
#                      axis=-1), \
#                  0, np.dot(qn_vec[-6], qn_vec[-6]), axis=-1))

#              qn_avg_row.append(np.insert( np.insert( \
#                    qn_vec[-3:], 
#                      0, np.dot(qn_vec[1], qn_vec[1]), axis=-1), \
#                        0, np.dot(qn_vec[0], qn_vec[0]), axis=-1))

            qn_avg_row = qn_row[0][-5:]
  
            qn_avg_row = np.asarray(qn_avg_row)
            qn_avg_gevp_col.append(qn_avg_row)
          qn_avg_gevp_col = np.asarray(qn_avg_gevp_col)
          qn_avg_gevp_row.append(qn_avg_gevp_col)
        qn_avg_gevp_row = np.asarray(qn_avg_gevp_row)
        qn_avg_irrep.append(qn_avg_gevp_row)
      qn_avg_irrep = np.asarray(qn_avg_irrep)
      qn_avg.append(qn_avg_irrep)
    qn_avg = np.asarray(qn_avg)
    np.save(path, qn_avg)
  
    print '\taveraging over rows'

    # write means over all rows of operators subducing into same irrep, [k1,k2]
    path = './bootdata/p%1i/%s_avg_rows' % (p, name)
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

    path = './bootdata/p%1i/%s_avg_rows_qn' % (p, name)
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
  
    for c in avg:
      print c.shape

################################################################################
# Bootstrap routine ############################################################

def bootstrap_ensemble(p, nb_bins, nb_boot, bootstrap_original_data):
  ################################################################################
  # read original data and call bootrap procedure
  diagram = 'C3+'

  if bootstrap_original_data:
    path = './readdata/p%1i/%s_p%1i.npy' % (p, diagram, p)
    data = np.load(path)
    path = './readdata/p%1i/%s_p%1i_quantum_numbers.npy' % (p, diagram, p)
    qn_data = np.load(path)
    if ( (qn_data.shape[0] != data.shape[0])):
      print '\tBootstrapped operators do not aggree with expected operators'
      exit(0)

    binned_data = prebinning(data.real, nb_bins)
    print 'Bootstrapping original data for p = %1i. Real part:' % p
    boot_real = bootstrap(binned_data, nb_boot, verbose=1)
    print boot_real.shape
    name = '%s_p%1i_real' % (diagram, p)
    write_ensemble(boot_real, qn_data, name, p)

    print 'Bootstrapping original data. Imaginary part:'
    binned_data = prebinning(data.imag, nb_bins)
    boot_imag = bootstrap(binned_data, nb_boot, verbose=1)
    print boot_imag.shape
    name = '%s_p%1i_imag' % (diagram, p)
    write_ensemble(boot_imag, qn_data, name, p)
   
    if (boot_real.shape[0] != boot_imag.shape[0]):
      print '\tSomething went wrong in splitting real from imaginary part.'
      exit(0)
    print '\tfinished bootstrapping original data'
  
  ################################################################################
  # read subduced data and call bootrap procedure
  path = './readdata/p%1i/%s_p%1i_subduced.npy' % (p, diagram, p)
  data = np.load(path)
  path = './readdata/p%1i/%s_p%1i_subduced_qn.npy' % (p, diagram, p)
  qn_subduced = np.load(path)
  if ( (qn_subduced.shape[0] != data.shape[0])):
    print 'Bootstrapped operators do not aggree with expected operators'
    exit(0)
  
  print 'Bootstrapping subduced data for p = %1i:' % p
  boot = []
  for irrep in data:
    boot_irrep = []
    for gevp_row in irrep:
      boot_gevp_row = []
      for gevp_col in gevp_row:
        boot_gevp_col = []
        for row in gevp_col:
          boot_row = []
          for k in row:
            binned_data = prebinning(k, nb_bins)
            boot_row.append(bootstrap(binned_data, nb_boot))
          boot_row = np.asarray(boot_row)
          boot_gevp_col.append(boot_row)
        boot_gevp_col = np.asarray(boot_gevp_col)
        boot_gevp_row.append(boot_gevp_col)
      boot_gevp_row = np.asarray(boot_gevp_row)
      boot_irrep.append(boot_gevp_row)
    boot_irrep = np.asarray(boot_irrep)
    boot.append(boot_irrep)
  boot = np.asarray(boot)
  print boot.shape
  name = '%s_p%1i_subduced' % (diagram, p)
  write_ensemble(boot, qn_subduced, name, p, True)

  print '\tfinished bootstrapping subduced operators'
  
#  ################################################################################
#  # Write data to disc ###########################################################
#  
#  ################################################################################
#  utils.ensure_dir('./bootdata')
#  utils.ensure_dir('./bootdata/p%1i' % p)
#  utils.ensure_dir('./bootdata/p%1i/single' % p)
#  utils.ensure_dir('./bootdata/p%1i/avg' % p)
#  
#  ################################################################################
#  # write original data if it was bootstraped before
#  
#  if bootstrap_original_data:
#    # write all operators
#    path = './bootdata/p%1i/%s_p%1i_real' % (p, diagram, p)
#    np.save(path, boot_real)
#    path = './bootdata/p%1i/%s_p%1i_imag' % (p, diagram, p)
#    np.save(path, boot_imag)
#    path = './bootdata/p%1i/%s_p%1i_qn' % (p, diagram, p)
#    np.save(path, qn_data)
#  
#  ################################################################################
#  # write all subduced correlators
#  
#  path = './bootdata/p%1i/%s_p%1i_subduced' % (p, diagram, p)
#  np.save(path, boot)
#  path = './bootdata/p%1i/%s_p%1i_subduced_qn' % (p, diagram, p)
#  np.save(path, qn_subduced)
#
#  # write means over all operators subducing into same irrep
#  avg = np.zeros_like(boot)
#  qn_avg = np.zeros_like(qn_subduced)
#  for i in range(boot.shape[0]):
#    for k in range(boot.shape[1]):
#      for g in range(boot.shape[2]):
#        for r in range(boot.shape[3]):
#          avg[i,k,g,r] = np.sum(boot[i,k,g,r], axis=0) 
#          qn_avg[i,k,g,r] = qn_subduced[i,k,g,r][0,3:]
#  avg = np.asarray(avg.tolist())
#  qn_avg = np.asarray(qn_avg.tolist())
#  path = './bootdata/p%1i/%s_p%1i_subduced_avg_vecks' % (p, diagram, p)
#  np.save(path, avg)
#  path = './bootdata/p%1i/%s_p%1i_subduced_avg_vecks_qn' % (p, diagram, p)
#  np.save(path, qn_avg)
#
#  avg = np.mean(avg, axis=-3)
#  path = './bootdata/p%1i/%s_p%1i_subduced_avg_rows' % (p, diagram, p)
#  np.save(path, avg)
#
#  qn_avg = qn_avg[...,0,:]
#  path = './bootdata/p%1i/%s_p%1i_subduced_avg_rows_qn' % (p, diagram, p)
#  np.save(path, qn_avg)

for p in range(0,2):
  bootstrap_ensemble(p, nb_bins, nb_boot, bootstrap_original_data)
