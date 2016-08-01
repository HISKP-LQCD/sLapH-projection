import h5py
import numpy as np
import itertools as it
import operator

import utils


def scalar_mul(x, y):
  return sum(it.imap(operator.mul, x, y))

def abs2(x):
  return scalar_mul(x, x)

def set_lookup_p(p_max, p_cm_max, p_cm, diagram):

  # create lookup table for all possible 3-momenta that can appear in our 
  # contractions
  lookup_p3 = list(it.ifilter(lambda x: abs2(x) <= p_max, \
                                 it.product(range(-p_max, p_max+1), repeat=3)))
  
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
  elif diagram == 'C3+':
    lookup_p = it.ifilter(lambda (w,x,y): \
               # if the sum of both momenta is the desired center of mass 
               # momentum and momentum conservation
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
  return list(lookup_p)

def set_qn(p, g, diagram):

  #contains list with momentum, displacement, gamma for both, source and sink
  if diagram == 'C20':
    return [p[0], np.zeros((3,)), np.asarray(g[0], dtype=int), \
            p[1], np.zeros((3,)), np.asarray(g[1], dtype=int)]
  elif diagram == 'C3+':
    return [p[0], np.zeros((3,)), np.asarray(5, dtype=int), \
            p[1], np.zeros((3,)), np.asarray(g, dtype=int), \
            p[2], np.zeros((3,)), np.asarray(5, dtype=int)]
  else:
    print 'in set_qn: diagram unknown! Quantum numbers corrupted.'
  return

def set_lookup_g(gammas, diagram):
  # all combinations, but only g1 with g01 etc. is wanted
  if diagram == 'C20':
    lookup_g = it.product([g for gamma in gammas for g in gamma[:-1]], repeat=2)
  elif diagram == 'C3+':
    lookup_g = it.product([g for gamma in gammas for g in gamma[:-1]], repeat=1)
  else:
    print 'in set_lookup_g: diagram unknown! Quantum numbers corrupted.'
    return
#  indices = [[1,2,3],[10,11,12],[13,14,15]]
#  lookup_g2 = [list(it.product([i[j] for i in indices], repeat=2)) for j in range(len(indices[0]))]
#  lookup_g = [item for sublist in lookup_g2 for item in sublist]

  return list(lookup_g)

def set_groupname(diagram, p, g):
  if diagram == 'C20':
    groupname = diagram + '_uu_p%1i%1i%1i.d000.g%i' % \
                                             (p[0][0], p[0][1], p[0][2], g[0]) \
                    + '_p%1i%1i%1i.d000.g%i' % (p[1][0], p[1][1], p[1][2], g[1])
  elif diagram == 'C3+':
    groupname = diagram + '_uuu_p%1i%1i%1i.d000.g5' % \
                                                   (p[0][0], p[0][1], p[0][2]) \
                        + '_p%1i%1i%1i.d000.g%1i' % \
                                                (p[1][0], p[1][1], p[1][2], g[0]) \
                        + '_p%1i%1i%1i.d000.g5' % (p[2][0], p[2][1], p[2][2])
  else:
    print 'in set_groupname: diagram unknown! Quantum numbers corrupted.'
    return

  return groupname

################################################################################
# reading configurations

def ensembles(sta_cnfg, end_cnfg, del_cnfg, diagram, p_cm, p_cm_max, p_max, gammas, T, directory, \
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

  lookup_p = set_lookup_p(p_max, p_cm_max, p_cm, diagram)
  lookup_g = set_lookup_g(gammas, diagram)
  
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

        groupname = set_groupname(diagram, p, g)
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


