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
def bootstrap(X, boot_size):
  np.random.seed(1227)
  boot = np.zeros(X[...,0].shape + (boot_size, ), dtype=float)
  length = X.shape[-1]
  # writing the mean value in the first sample
  boot[...,0] = np.mean(X, axis=-1) 
  # doing all other samples
  for i in range(1, boot_size): 
    if i % (boot_size/10) == 0:
      print '\tbootstrap sample %i' % i
    rnd = np.random.random_integers(0, high=length-1, size=length)
    boot[...,i] = np.mean(X[...,rnd], axis=-1)
  return boot

## bootstrap an arbitrary array with nb_cnfg as last dimension
#def bootstrap(X, boot_size):
#  np.random.seed(1227)
#  boot = np.zeros(X[:,:,0].shape + (boot_size, ), dtype=float)
#  length = X.shape[-1]
#  # writing the mean value in the first sample
#  boot[:,:,0] = np.mean(X, axis=-1) 
#  # doing all other samples
#  for i in range(1, boot_size): 
#    if i % (boot_size/10) == 0:
#      print '\tbootstrap sample %i' % i
#    rnd = np.random.random_integers(0, high=length-1, size=length)
#    boot[:,:,i] = np.mean(X[:,:,rnd], axis=-1)
#  return boot

################################################################################
# Bootstrap routine ############################################################

def bootstrap_ensembles(p, nb_boot, bootstrap_original_data):
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

    binned_data = prebinning(data.real, 5)
    print 'Bootstrapping original data for p = %1i. Real part:' % p
    boot_real = bootstrap(binned_data, nb_boot)
    print 'Bootstrapping original data. Imaginary part:'
    binned_data = prebinning(data.imag, 1)
    boot_imag = bootstrap(binned_data, nb_boot)
    print boot_real.shape
    
    if (boot_real.shape[0] != boot_imag.shape[0]):
      print '\tSomething went wrong in splitting real from imaginary part.'
      exit(0)
    print '\tfinished bootstrapping original data'
  
  ################################################################################
  # read subduced data and call bootrap procedure
  path = './readdata/p%1i/%s_p%1i_single_subduced.npy' % (p, diagram, p)
  data = np.load(path)
  path = './readdata/p%1i/%s_p%1i_single_subduced_quantum_numbers.npy' % (p, diagram, p)
  qn_subduced = np.load(path)
  if ( (qn_subduced.shape[0] != data.shape[0])):
    print 'Bootstrapped operators do not aggree with expected operators'
    exit(0)
  
  print 'Bootstrapping subduced data:'
  boot = []
  for irrep in data:
    for gamma in irrep:
      for mom in gamma:
        for row in mom:
          binned_data = prebinning(row, 1)
          boot.append(bootstrap(binned_data, nb_boot))
  boot = np.asarray(boot).reshape(data.shape)
  print boot.shape
  print '\tfinished bootstrapping subduced operators'
  
  ################################################################################
  # Write data to disc ###########################################################
  
  ################################################################################
  utils.ensure_dir('./bootdata')
  utils.ensure_dir('./bootdata/p%1i' % p)
  utils.ensure_dir('./bootdata/p%1i/single' % p)
  utils.ensure_dir('./bootdata/p%1i/avg' % p)
  
  ################################################################################
  # write original data if it was bootstraped before
  
  if bootstrap_original_data:
    # write all operators
    path = './bootdata/p%1i/%s_p%1i_real' % (p, diagram, p)
    np.save(path, boot_real)
    path = './bootdata/p%1i/%s_p%1i_imag' % (p, diagram, p)
    np.save(path, boot_imag)
    path = './bootdata/p%1i/%s_p%1i_quantum_numbers' % (p, diagram, p)
    np.save(path, qn_data)
  
    # write every operator seperately
#    for i in range(0, qn_data.shape[0]):
#      path = './bootdata/p%1i/single/C20_single_p%i%i%i.d%i%i%i.g%i_' \
#             'p%i%i%i.d%i%i%i.g%i_real' % \
#              (p, qn_data[i][0][0], qn_data[i][0][1], qn_data[i][0][2], \
#               qn_data[i][1][0], qn_data[i][1][1], qn_data[i][1][2], \
#               qn_data[i][2], qn_data[i][3][0], qn_data[i][3][1], \
#               qn_data[i][3][2], qn_data[i][4][0], qn_data[i][4][1], \
#               qn_data[i][4][2], qn_data[i][5])
#      np.save(path, boot_real[i])
#      path = './bootdata/p%1i/single/C20_single_p%i%i%i.d%i%i%i.g%i_' \
#             'p%i%i%i.d%i%i%i.g%i_imag' % \
#              (p, qn_data[i][0][0], qn_data[i][0][1], qn_data[i][0][2], \
#               qn_data[i][1][0], qn_data[i][1][1], qn_data[i][1][2], \
#               qn_data[i][2], qn_data[i][3][0], qn_data[i][3][1], \
#               qn_data[i][3][2], qn_data[i][4][0], qn_data[i][4][1], \
#               qn_data[i][4][2], qn_data[i][5])
#      np.save(path, boot_imag[i])
   
  ################################################################################
  # write all subduced correlators
  
  path = './bootdata/p%1i/%s_p%1i_subduced' % (p, diagram, p)
  np.save(path, boot)
  path = './bootdata/p%1i/%s_p%1i_subduced_quantum_numbers' % (p, diagram, p)
  np.save(path, qn_subduced)

  # write means over all operators subducing into same irrep
  avg = np.zeros_like(boot)
  qn_avg = np.zeros_like(qn_subduced)
  for i in range(boot.shape[0]):
    for g in range(boot.shape[1]):
      for k in range(boot.shape[2]):
        for r in range(boot.shape[3]):
          avg[i,g,k,r] = np.sum(boot[i,g,k,r], axis=0) 
          qn_avg[i,g,k,r] = qn_subduced[i,g,k,r][0,3:]
  avg = np.asarray(avg.tolist())
  qn_avg = np.asarray(qn_avg.tolist())
  path = './bootdata/p%1i/%s_p%1i_subduced_avg_vecks' % (p, diagram, p)
  np.save(path, avg)
  path = './bootdata/p%1i/%s_p%1i_subduced_avg_vecks_quantum_numbers' % (p, diagram, p)
  np.save(path, qn_avg)

  avg = np.mean(avg, axis=-3)
  path = './bootdata/p%1i/%s_p%1i_subduced_avg_rows' % (p, diagram, p)
  np.save(path, avg)

  qn_avg = qn_avg[...,0,:]
  path = './bootdata/p%1i/%s_p%1i_subduced_avg_rows_quantum_numbers' % (p, diagram, p)
  np.save(path, qn_avg)

  
#  # write means over all operators subducing into same irrep
#  if p not in [1,3,4]:
#    path = './bootdata/p%1i/C20_p%1i_avg_subduced' % (p, p)
#    np.save(path, np.mean(boot, axis=2) )
#    path = './bootdata/p%1i/C20_p%1i_avg_subduced_quantum_numbers' % (p, p)
#    np.save(path, qn_subduced[...,0,-3:])
#  else:
#    # if p in [1,3,4], the 2dim E2 irrep appears. Both rows can be averaged, thus
#    # the average is taken and appended in the end for plotting
#    path = './bootdata/p%1i/C20_p%1i_avg_subduced' % (p, p)
#    E2 = np.zeros((1,) + np.mean(boot, axis=2)[0].shape)
#    for j in range(0,boot.shape[1]):
#      E2[0,j] = np.mean(np.vstack((boot[1,j], boot[2,j])), axis=0)
#    np.save(path, np.vstack((np.mean(boot, axis=2), E2 )) )
#    path = './bootdata/p%1i/C20_p%1i_avg_subduced_quantum_numbers' % (p, p)
#    E2_qn = np.zeros((1,qn_subduced.shape[1], qn_subduced[...,-3:].shape[-1]), \
#                                                                  dtype=((str,256)))
#    E2_qn[0] = qn_subduced[0,:,0,-3:]
#    for j in range(0,boot.shape[1]):
#      E2_qn[0,j,-1] = 'E2'
#    np.save(path, np.vstack((qn_subduced[...,0,-3:], E2_qn)) )
#  
#  ################################################################################
#  # write the subduced correlators for each irrep and gamma seperately
#  for i in ( range(0, boot.shape[0]) ):
#    for j in ( range(0, boot.shape[1]) ):
#      path = './bootdata/p%1i/single/C20_p%1i_single_%s_%s_%s' % (p, p, \
#              qn_subduced[i,j,0,-3], qn_subduced[i,j,0,-2], qn_subduced[i,j,0,-1])
#      np.save(path, boot[i,j])
#      path = './bootdata/p%1i/single/C20_p%1i_single_%s_%s_%s_quantum_numbers' % \
#              (p, p, qn_subduced[i,j,0,-3], qn_subduced[i,j,0,-2], 
#                                                            qn_subduced[i,j,0,-1])
#      np.save(path, qn_subduced[i,j])
#  
#  # write means over all operators subducing into same irrep for each irrep and 
#  # gamma seperately
#  for i in ( range(0, boot.shape[0]) ):
#    for j in ( range(0, boot.shape[1]) ):
#      path = './bootdata/p%1i/avg/C20_p%1i_avg_%s_%s_%s' % (p, p, \
#              qn_subduced[i,j,0,-3], qn_subduced[i,j,0,-2], qn_subduced[i,j,0,-1])
#      np.save(path, np.mean(boot[i,j], axis=0) )
#      path = './bootdata/p%1i/avg/C20_p%1i_avg_%s_%s_%s_quantum_numbers' % (p, p, \
#              qn_subduced[i,j,0,-3], qn_subduced[i,j,0,-2], qn_subduced[i,j,0,-1])
#      np.save(path, qn_subduced[i,j,0,-1])
#  
#  print '\tfinished writing'

bootstrap_ensembles(0, 500, False)
