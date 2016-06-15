#!/usr/bin/python

import numpy as np

import sys

import utils
#sys.stdout = open('./boot.out', 'w')

################################################################################
# Parameters ###################################################################

################################################################################
bootstrap_original_data = False

p = range(1,2)         # momentum

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
def bootstrap(X, boot_size, write=False):
  np.random.seed(1227)
  boot = np.zeros(X[...,0].shape + (boot_size, ), dtype=float)
  length = X.shape[-1]
  # writing the mean value in the first sample
  boot[...,0] = np.mean(X, axis=-1) 
  # doing all other samples
  for i in range(1, boot_size): 
    if i % (boot_size/10) == 0:
      if write == True:
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

def bootstrap_ensembles(p_cm, nb_bins, nb_boot, bootstrap_original_data):
  ################################################################################
  # read original data and call bootrap procedure
  diagram = 'C4'

  if bootstrap_original_data:
    path = './readdata/p%1i/%s_p%1i.npy' % (p_cm, diagram, p_cm)
    data = np.load(path)
    path = './readdata/p%1i/%s_p%1i_quantum_numbers.npy' % (p_cm, diagram, p_cm)
    qn_data = np.load(path)
    if ( (qn_data.shape[0] != data.shape[0])):
      print '\tBootstrapped operators do not aggree with expected operators'
      exit(0)

    binned_data = prebinning(data.real, nb_bins)
    print 'Bootstrapping original data for p = %1i. Real part:' % p_cm
    boot_real = bootstrap(binned_data, nb_boot, write=True)
    print 'Bootstrapping original data. Imaginary part:'
    binned_data = prebinning(data.imag, nb_bins)
    boot_imag = bootstrap(binned_data, nb_boot, write=True)
    print boot_real.shape
    
    if (boot_real.shape[0] != boot_imag.shape[0]):
      print '\tSomething went wrong in splitting real from imaginary part.'
      exit(0)
    print '\tfinished bootstrapping original data'
  
  ################################################################################
  # read subduced data and call bootrap procedure
  path = './readdata/p%1i/%s_p%1i_subduced.npy' % (p_cm, diagram, p_cm)
  data = np.load(path)
  path = './readdata/p%1i/%s_p%1i_subduced_qn.npy' % (p_cm, diagram, p_cm)
  qn_subduced = np.load(path)
  if ( (qn_subduced.shape[0] != data.shape[0])):
    print 'Bootstrapped operators do not aggree with expected operators'
    exit(0)
  
#  print 'Bootstrapping subduced data:'
#  boot = []
#  for irrep in data:
#    for gevp_row in irrep:
#      for gevp_col in gevp_row:
#        for row in gevp_col:
#          binned_data = prebinning(row, nb_bins)
#          boot.append(bootstrap(binned_data, nb_boot))
#  boot = np.asarray(boot).reshape(data.shape)
  print 'Bootstrapping subduced data for p = %1i:' % p_cm
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
  print '\tfinished bootstrapping subduced operators'
  
  ################################################################################
  # Write data to disc ###########################################################
  
  ################################################################################
  utils.ensure_dir('./bootdata')
  utils.ensure_dir('./bootdata/p%1i' % p_cm)
  utils.ensure_dir('./bootdata/p%1i/single' % p_cm)
  utils.ensure_dir('./bootdata/p%1i/avg' % p_cm)
  
  ################################################################################
  # write original data if it was bootstraped before
  
  if bootstrap_original_data:
    # write all operators
    path = './bootdata/p%1i/%s_p%1i_real' % (p_cm, diagram, p_cm)
    np.save(path, boot_real)
    path = './bootdata/p%1i/%s_p%1i_imag' % (p_cm, diagram, p_cm)
    np.save(path, boot_imag)
    path = './bootdata/p%1i/%s_p%1i_quantum_numbers' % (p_cm, diagram, p_cm)
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
  
  path = './bootdata/p%1i/%s_p%1i_subduced' % (p_cm, diagram, p_cm)
  np.save(path, boot)
  path = './bootdata/p%1i/%s_p%1i_subduced_qn' % (p_cm, diagram, p_cm)
  np.save(path, qn_subduced)

  # write means over all operators subducing into same irrep

#  avg = np.zeros_like(boot)
#  qn_avg = np.zeros_like(qn_subduced)
  # recursive if shape[-2] != T: for i in np.ndindex(): boot else: boot
  # recursive if not double but np.array: for i in np.ndindex(): boot else: boot
#  for i in range(boot.shape[0]):
#    for k1 in range(boot.shape[1]):
#      for k2 in range(boot.shape[2]):
#        for r in range(boot.shape[3]):
#          avg[i,k1,k2,r] = np.sum(boot[i,k1,k2,r], axis=0) 
#          qn_avg[i,k1,k2,r] = qn_subduced[i,k1,k2,r][0,4:]
#  avg = np.asarray(avg.tolist())
#  print avg.shape
#  qn_avg = np.asarray(qn_avg.tolist())
  print '\taveraging over momenta'
  path = './bootdata/p%1i/%s_p%1i_subduced_avg_vecks' % (p_cm, diagram, p_cm)
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

  path = './bootdata/p%1i/%s_p%1i_subduced_avg_vecks_qn' % (p_cm, diagram, p_cm)
  qn_avg = []
  for i, qn_irrep in enumerate(qn_subduced):
    qn_avg_irrep = []
    for g1, qn_gevp_row in enumerate(qn_irrep):
      qn_avg_gevp_row = []
      for g2, qn_gevp_col in enumerate(qn_gevp_row):
        qn_avg_gevp_col = []
        for r, qn_row in enumerate(qn_gevp_col):
          qn_avg_gevp_col.append(np.asarray(qn_row[0][-5:]))
        qn_avg_gevp_col = np.asarray(qn_avg_gevp_col)
        qn_avg_gevp_row.append(qn_avg_gevp_col)
      qn_avg_gevp_row = np.asarray(qn_avg_gevp_row)
      qn_avg_irrep.append(qn_avg_gevp_row)
    qn_avg_irrep = np.asarray(qn_avg_irrep)
    qn_avg.append(qn_avg_irrep)
  qn_avg = np.asarray(qn_avg)
  np.save(path, qn_avg)

  #write means over all operators subducing into same irrep
  print '\taveraging over rows'
  path = './bootdata/p%1i/%s_p%1i_subduced_avg_rows' % (p_cm, diagram, p_cm)
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
  for c in avg:
    print c.shape
  np.save(path, avg)

  path = './bootdata/p%1i/%s_p%1i_subduced_avg_rows_qn' % (p_cm, diagram, p_cm)
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

#  avg = np.mean(avg, axis=-3)
#  print avg.shape
#  np.save(path, avg)
#  qn_avg = qn_avg[...,0,:]
#  np.save(path, qn_avg)

  
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

for p_cm in p:
  bootstrap_ensembles(p_cm, nb_bins, nb_boot, bootstrap_original_data)
