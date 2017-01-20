import h5py
import numpy as np
import itertools as it
import operator

from pandas import Series, DataFrame
import pandas as pd

import utils


def scalar_mul(x, y):
  return sum(it.imap(operator.mul, x, y))

def abs2(x):
  return scalar_mul(x, x)

# TODO: nb_cnfg is spurious, can just use len(lookup_cnfg)
def set_lookup_cnfg(sta_cnfg, end_cnfg, del_cnfg, missing_configs, verbose=0):
  """
  Get a list of all gauge configurations contractions where performed on

  Parameters
  ----------
  sta_cnfg : int
      Number of first gauge configuration.
  end_cnfg : int
      Number of last gauge configuration.
  del_cnfg : int
      Step size chosen between following gauge configurations.
  missing_configs : list of int
      List of configurations to be omitted because the contractions were not
      performed

  Returns
  -------
  lookup_cnfg : list of int
      List of the configurations to read
  """

  # calculate number of configurations
  nb_cnfg = 0
  lookup_cnfg = []
  for cnfg in range(sta_cnfg, end_cnfg+1, del_cnfg):
    if cnfg in missing_configs:
      continue
    lookup_cnfg.append(cnfg)
  if(verbose):
    print 'number of configurations: %i' % len(lookup_cnfg)

  return lookup_cnfg


def set_lookup_p(p_max, p_cm, diagram):

  # create lookup table for all possible 3-momenta that can appear in our 
  # contractions
  # depending on chosen diagram create allowed momenta or combinations of two
  # momenta at source/sink and then combine them demanding momentum conservation

  # still need to cutoff at momentum 2 for p_cm==0 and check whether the additional
  # momenta for p_cm==4 can be used

  # TODO: write wrapper function that calculates lookup_p3 and make that a parameter itself
  # TODO: generalize lookup_p construction by writing a function that can get either a momentum
  # or a tuple made of two
  lookup_p3 = list(it.ifilter(lambda x: abs2(x) <= p_max, \
                                  it.product(range(-p_max, p_max+1), repeat=3)))
  lookup_p3_reduced = [(0,0,0), (0,0,1), (0,1,1), (1,1,1), (0,0,2)]
  
  if diagram == 'C20':
    lookup_so = it.ifilter(lambda x : abs2(x) == p_cm, lookup_p3)
    lookup_so, lookup_si = it.tee(lookup_so, 2)
    lookup_p = it.ifilter(lambda (x,y): \
                            tuple(x) == tuple(it.imap(operator.neg, y)), \
                                               it.product(lookup_so, lookup_si))
  elif diagram == 'C3+':
    lookup_so = it.ifilter(lambda (x,y): \
                             abs2(list(it.imap(operator.add, x, y))) == p_cm and \
                             not (p_cm == 0 and (tuple(x) == tuple(y))), \
                                                it.product(lookup_p3, repeat=2))
    lookup_si = it.ifilter(lambda x : abs2(x) == p_cm, lookup_p3)
    lookup_p = it.ifilter(lambda ((w,x),y): \
                            tuple(it.imap(operator.add, w, x)) == tuple(it.imap(operator.neg, y)), \
                                               it.product(lookup_so, lookup_si))

  elif diagram.startswith('C4'):
    lookup_so = it.ifilter(lambda (x,y): \
#                             abs2(list(it.imap(operator.add, x, y))) == p_cm and \
                             (tuple(it.imap(operator.add, x, y)) == lookup_p3_reduced[p_cm]) and \
                             not (p_cm == 0 and (tuple(x) == tuple(y))), \
                                                it.product(lookup_p3, repeat=2))
    lookup_si = it.ifilter(lambda (x,y): \
                             (tuple(it.imap(operator.neg, it.imap(operator.add, x, y))) == lookup_p3_reduced[p_cm]) and \
                             not (p_cm == 0 and (tuple(x) == tuple(y))), \
                                                it.product(lookup_p3, repeat=2))
#    lookup_so, lookup_si = it.tee(lookup_so, 2)
    lookup_p = it.ifilter(lambda ((w,x),(y,z)): \
                            tuple(it.imap(operator.add, w, x)) == tuple(it.imap(operator.neg, it.imap(operator.add, y, z))), \
                                               it.product(lookup_so, lookup_si))
  else:
    print 'in set_lookup_p: diagram unknown! Quantum numbers corrupted.'
  return list(lookup_p)

def set_lookup_g(gammas, diagram):
  # TODO: currently calculates all combinations, but only g1 with g01 etc. is wanted,
  # not eg. g1 with g02
  # TODO: gamma_5 is hardcoded. That should be generelized in the future
  # TODO: combine set_lookup_p and set_lookup_g (and set_lookup_d) to possible set_lookup_qn?

  if diagram == 'C20':
    lookup_so = it.product([g for gamma in gammas for g in gamma[:-1]])
    lookup_so, lookup_si = it.tee(lookup_so, 2)
  elif diagram == 'C3+':
    lookup_so = it.product([5], [5]) 
    lookup_si = it.product([g for gamma in gammas for g in gamma[:-1]])
  elif diagram.startswith('C4'):
    lookup_so = it.product([5], [5]) 
    lookup_so, lookup_si = it.tee(lookup_so, 2)
  else:
    print 'in set_lookup_g: diagram unknown! Quantum numbers corrupted.'
    return
#  indices = [[1,2,3],[10,11,12],[13,14,15]]
#  lookup_g2 = [list(it.product([i[j] for i in indices], repeat=2)) for j in range(len(indices[0]))]
#  lookup_g = [item for sublist in lookup_g2 for item in sublist]

  lookup_g = it.product(lookup_so, lookup_si)
  return list(lookup_g)

def set_lookup_qn(diagram, p_cm, p_max, gammas, verbose=0):
  """
  Calculates a data frame with physical quantum numbers

  Parameters
  ----------
  diagram : string, {'C20', 'C3+', 'C4+B', 'C4+D'}
      Diagram of wick contractions for the rho meson.
  p_cm : int, {0, 1, 2, 3, 4}
      Center of mass momentum.
  p_max : int
      Maximum entry of momentum vectors.
  gammas : list of list of ints and string
      A list which for each gamma structure coupling to the rho meson contains
      a list with the integer indices used in the contraction code and a 
      latex-style name for plotting labels.

  Returns
  -------
  lookup_qn : pd.DataFrame
      pandas DataFrame where each row is a combination of quantum numbers and
      the row index is used as identifier for it.
  """

  lookup_p = set_lookup_p(p_max, p_cm, diagram)
  lookup_g = set_lookup_g(gammas, diagram)

  # TODO: A more elegant solution for combining lookup_p and lookup_g is welcome
  # maybe Multiindex.from_product()
  tmp = it.product(lookup_p, lookup_g)
  lookup_qn = []
  for t in tmp:
    lookup_qn.append(t[0]+t[1])
  lookup_qn = DataFrame(lookup_qn, columns=['p_{so}', 'p_{si}', '\gamma_{so}', '\gamma_{si}'])
#  lookup_qn['p_{so}'] = qn['p_{so}'].apply(np.array)
#  lookup_qn['p_{si}'] = qn['p_{si}'].apply(np.array)
  
  return lookup_qn


def set_groupname(diagram, p, g):
  # sets groupname the desired correlator in the hdf5 output of the cntrv0.1-code.
  # TODO: displacement hardcoded
  if diagram == 'C20':
    groupname = diagram + '_uu_p%1i%1i%1i.d000.g%i' % \
                                             (p[0][0], p[0][1], p[0][2], g[0][0]) \
                    + '_p%1i%1i%1i.d000.g%i' % (p[1][0], p[1][1], p[1][2], g[1][0])
  elif diagram == 'C3+':
    groupname = diagram + '_uuu_p%1i%1i%1i.d000.g5' % \
                                                   (p[0][0][0], p[0][0][1], p[0][0][2]) \
                        + '_p%1i%1i%1i.d000.g%1i' % \
                                                (p[1][0], p[1][1], p[1][2], g[1][0]) \
                        + '_p%1i%1i%1i.d000.g5' % (p[0][1][0], p[0][1][1], p[0][1][2])
  elif diagram == 'C4+D':
    groupname = diagram + '_uuuu_p%1i%1i%1i.d000.g5' % (p[0][0][0], p[0][0][1], p[0][0][2]) + \
             '_p%1i%1i%1i.d000.g5' % (p[1][0][0], p[1][0][1], p[1][0][2]) + \
             '_p%1i%1i%1i.d000.g5' % (p[0][1][0], p[0][1][1], p[0][1][2]) + \
             '_p%1i%1i%1i.d000.g5' % (p[1][1][0], p[1][1][1], p[1][1][2])
  elif diagram == 'C4+B':
    groupname = diagram + '_uuuu_p%1i%1i%1i.d000.g5' % (p[0][0][0], p[0][0][1], p[0][0][2]) + \
             '_p%1i%1i%1i.d000.g5' % (p[1][0][0], p[1][0][1], p[1][0][2]) + \
             '_p%1i%1i%1i.d000.g5' % (p[1][1][0], p[1][1][1], p[1][1][2]) + \
             '_p%1i%1i%1i.d000.g5' % (p[0][1][0], p[0][1][1], p[0][1][2]) 

  else:
    print 'in set_groupname: diagram unknown! Quantum numbers corrupted.'
    return

  return groupname

################################################################################
# reading configurations

def ensembles(sta_cnfg, end_cnfg, del_cnfg, diagram, p_cm, p_cm_max, p_max, gammas, T, directory, \
                                                    missing_configs, verbose=0):
  
  print 'reading data for %s, p=%i' % (diagram, p_cm)

  lookup_cnfg = set_lookup_cnfg(sta_cnfg, end_cnfg, del_cnfg, \
                                                       missing_configs, verbose)

  # set up lookup table for quantum numbers
  if p_cm == 0:
    p_max = 2

  lookup_qn = set_lookup_qn(diagram, p_cm, p_max, gammas, verbose)

  data = []

  for cnfg in lookup_cnfg:
    # filename and path
    filename = directory + 'cnfg%i/' % cnfg + diagram + '_cnfg%i' % cnfg + '.h5'

    if verbose:
      print filename

    # to achieve hirarchical indexing for quantum numbers build DataFrame for
    # each loop seperately
    # TODO: is it necessary to builed that completely or can that be 
    # constructed by successively storing each operator with pd.HDFStore()?
    data_qn = DataFrame()
#    print DataFrame(lookup_p)
#    print DataFrame(lookup_g)

    for op in lookup_qn.index:
      p = lookup_qn.ix[op, ['p_{so}', 'p_{si}']]
      g = lookup_qn.ix[op, ['\gamma_{so}', '\gamma_{si}']]
      groupname = set_groupname(diagram, p, g)
      if verbose:
        print groupname
      # TODO: catch when a groupname does not exist

      data_qn[op] = pd.read_hdf(filename, key=groupname).stack()
    data.append(data_qn)
  data = pd.concat(data, keys=lookup_cnfg, axis=0)
 
  print '\tfinished reading'
  
  ################################################################################
  # write data to disc
  
  utils.ensure_dir('./readdata')

  store = pd.HDFStore('./readdata/%s_p%1i.h5' % (diagram, p_cm))
  # write all operators
  store['data'] = data
  store['qn'] = lookup_qn

  store.close()
  
  print '\tfinished writing'

