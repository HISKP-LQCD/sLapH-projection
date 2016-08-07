import numpy as np

import utils
# add all relevant diagrams with according Clebsch-Gordans

gamma_i =   [1, 2, 3, 'gi']
gamma_0i =  [10, 11, 12, 'g0gi']
gamma_50i = [13, 14, 15, 'g5g0gi']

gammas = [gamma_i, gamma_0i, gamma_50i]

def rho_2pt(p_cm, diagram='C20', verbose=0):
  wickd = []
  qn_wickd = []
  
  data = []
  qn_data = []
  path = './readdata/p%1i/%s_p%1i.npy' % (p_cm, diagram, p_cm)
  data = np.load(path)
  path = './readdata/p%1i/%s_p%1i_qn.npy' % (p_cm, diagram, p_cm)
  qn_data = np.load(path)
  if ( (qn_data.shape[0] != data.shape[0])):
    print '\tBootstrapped operators do not aggree with expected operators'
    exit(0)
  
  if verbose:
    print qn_data.shape
  
  for i, qn in enumerate(qn_data):
   # if source momenta equal
    if verbose: 
      print i, qn[0], '-g', qn[2], qn[3], '-g', qn[5]

    # multiply diagrams
    if ((qn[2] in gamma_i) and (qn[5] in gamma_0i)) \
      or ((qn[2] in gamma_0i) and (qn[5] in gamma_0i)):
      wick_factor = -2.
    elif ((qn[2] in gamma_i) and (qn[5] in gamma_50i)) \
      or ((qn[2] in gamma_50i) and (qn[5] in gamma_i)) \
      or ((qn[2] in gamma_0i) and (qn[5] in gamma_50i)):
      wick_factor = 2. * 1j
    elif ((qn[2] in gamma_50i) and (qn[5] in gamma_0i)):
      wick_factor = -2. * 1j
    elif ((qn[2] in gamma_0i) and (qn[5] in gamma_i)) \
      or ((qn[2] in gamma_i) and (qn[5] in gamma_i)) \
      or ((qn[2] in gamma_50i) and (qn[5] in gamma_50i)):
      # case diagonal elements or g0gi-gi
      wick_factor = 2.
    else:
      continue

    wickd.append(wick_factor*data[i])
    qn_wickd.append(qn)
  
  wickd = np.asarray(wickd)
  print wickd.shape
  qn_wickd = np.asarray(qn_wickd)
  print qn_wickd.shape
  
  ################################################################################
  # write data to disc
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i' % p_cm)

  # write every operator seperately
  #for i in range(0, qn_wickd.shape[0]):
  #  path = './readdata/p%1i/single/%s/%s' % \
  #          (p, 'C4', qn_wickd[i][-1])
  #  np.save(path, wickd[i])
  
  # write all operators
  path = './readdata/p%1i/%s_p%1i' % (p_cm, 'C2', p_cm)
  np.save(path, wickd)
  
  # write all quantum numbers
  path = './readdata/p%1i/%s_p%1i_qn' % (p_cm, 'C2', p_cm)
  np.save(path, qn_wickd)
  
  print '\tfinished writing\n'

  ################################################################################

def rho_3pt(p_cm, diagram='C3+', verbose=0):
  wickd = []
  qn_wickd = []
  
  data = []
  qn_data = []
  path = './readdata/p%1i/%s_p%1i.npy' % (p_cm, diagram, p_cm)
  data = np.load(path)
  path = './readdata/p%1i/%s_p%1i_qn.npy' % (p_cm, diagram, p_cm)
  qn_data = np.load(path)
  if ( (qn_data.shape[0] != data.shape[0])):
    print '\tBootstrapped operators do not aggree with expected operators'
    exit(0)
  
  if verbose:
    print qn_data.shape
  
  for i, qn in enumerate(qn_data):
   # if source momenta equal
    if verbose: 
      print i, qn[0], '-g', qn[2], qn[3], '-g', qn[5], qn[6], '-g', qn[8]

    # multiply diagrams
    if qn[5] in gamma_i:
      wick_factor = 2.
    elif qn[5] in gamma_0i:
      wick_factor = -2.
    elif qn[5] in gamma_50i:
      wick_factor = 2*1j
    else:
      continue

    wickd.append(wick_factor*data[i])
    qn_wickd.append(qn)
  
  wickd = np.asarray(wickd)
  print wickd.shape
  qn_wickd = np.asarray(qn_wickd)
  print qn_wickd.shape
  
  ################################################################################
  # write data to disc
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i' % p_cm)

  # write every operator seperately
  #for i in range(0, qn_wickd.shape[0]):
  #  path = './readdata/p%1i/single/%s/%s' % \
  #          (p, 'C4', qn_wickd[i][-1])
  #  np.save(path, wickd[i])
  
  # write all operators
  path = './readdata/p%1i/%s_p%1i' % (p_cm, 'C3', p_cm)
  np.save(path, wickd)
  
  # write all quantum numbers
  path = './readdata/p%1i/%s_p%1i_qn' % (p_cm, 'C3', p_cm)
  np.save(path, qn_wickd)
  
  print '\tfinished writing\n'

  ################################################################################

def rho_4pt(p_cm, diagrams, verbose=0):
  wickd = []
  qn_wickd = []
  
  data = []
  qn_data = []
  for diagram in diagrams:
    path = './readdata/p%1i/%s_p%1i.npy' % (p_cm, diagram, p_cm)
    data.append(np.load(path))
    path = './readdata/p%1i/%s_p%1i_qn.npy' % (p_cm, diagram, p_cm)
    qn_data.append(np.load(path))
    if ( (qn_data[-1].shape[0] != data[-1].shape[0])):
      print '\tBootstrapped operators do not aggree with expected operators'
      exit(0)
  qn_data = np.asarray(qn_data)
  
  if verbose:
    print qn_data.shape
  
  for i, qn_box in enumerate(qn_data[0]):
    for j, qn_dia in enumerate(qn_data[1]):
      # if source momenta equal
      if (qn_box[0] == qn_dia[0] and qn_box[3] == qn_dia[3]):
        # if sink momenta equal
        if (qn_box[6] == qn_dia[6] and qn_box[9] == qn_dia[9]):
          if verbose: 
            print i, j, qn_box[0], qn_box[3], qn_box[6], qn_box[9]
          # add diagrams
          wickd.append((-2)*data[0][i]+1*data[1][j])
          qn_wickd.append([qn_box[0], qn_box[1], qn_box[2], \
                           qn_box[3], qn_box[4], qn_box[5], \
                           qn_box[6], qn_box[7], qn_box[8], \
                           qn_box[9], qn_box[10], qn_box[11]])
  
  wickd = np.asarray(wickd)
  print wickd.shape
  qn_wickd = np.asarray(qn_wickd)
  print qn_wickd.shape
  
  ################################################################################
  # write data to disc
  
  utils.ensure_dir('./readdata')
  utils.ensure_dir('./readdata/p%1i' % p_cm)

  # write every operator seperately
  #for i in range(0, qn_wickd.shape[0]):
  #  path = './readdata/p%1i/single/%s/%s' % \
  #          (p, 'C4', qn_wickd[i][-1])
  #  np.save(path, wickd[i])
  
  # write all operators
  path = './readdata/p%1i/%s_p%1i' % (p_cm, 'C4', p_cm)
  np.save(path, wickd)
  
  # write all quantum numbers
  path = './readdata/p%1i/%s_p%1i_qn' % (p_cm, 'C4', p_cm)
  np.save(path, qn_wickd)
  
  print '\tfinished writing\n'
 
