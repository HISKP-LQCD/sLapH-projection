# functions for munging and cleaning correlators

import h5py
import numpy as np
import itertools as it
import operator

from pandas import Series, DataFrame
import pandas as pd

def _scalar_mul(x, y):
  return sum(it.imap(operator.mul, x, y))

def _abs2(x):
  return _scalar_mul(x, x)

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
    print '\t\tNumber of configurations: %i' % len(lookup_cnfg)

  return lookup_cnfg


# TODO: write wrapper function that calculates lookup_p3 and make that a 
# parameter itself
# TODO: generalize lookup_p construction by writing a function that can get 
# either a momentum or a tuple made of two
# TODO: 4pt function is incosistent for box and direct diagram
def set_lookup_p(p_max, p_cm, diagram):
  """
  create lookup table for all possible 3-momenta that can appear on the lattice
  below a given cutoff

  Parameters
  ----------
  p_max : int
      Cut-off. Do not consider momenta with a higher absolute value than this
  p_cm : int
      Absolute value of sum of all momenta at source or sink. Must be equal at
      both due to momentum conservation
  diagram : string, {'C20', 'C2+', 'C3+', 'C4*'}
      The number of momenta is equal to the number of quarks in a chosen 
      diagram. It is encoded in the second char of the diagram name

  Returns
  -------
  list of tuples (of tuples)
      List that contains tuples for source and sink with tuples for the momenta.
      For every possible momentum combination, there is one list entry
  """

  # for the center-of-mass frame p_max was restricted to (1,1,0)
  if p_cm == 0:
    p_max = 2

  lookup_p3 = list(it.ifilter(lambda x: _abs2(x) <= p_max, \
                                  it.product(range(-p_max, p_max+1), repeat=3)))
  lookup_p3_reduced = [(0,0,0), (0,0,1), (0,1,1), (1,1,1), (0,0,2)]
  
  if diagram == 'C20' or diagram == 'C2+':
    lookup_so = it.ifilter(lambda x : _abs2(x) == p_cm, lookup_p3)
    lookup_so, lookup_si = it.tee(lookup_so, 2)
    lookup_p = it.ifilter(lambda (x,y): \
                            tuple(x) == tuple(it.imap(operator.neg, y)), \
                                               it.product(lookup_so, lookup_si))
  elif diagram == 'C3+':
    lookup_so = it.ifilter(lambda (x,y): \
                             _abs2(list(it.imap(operator.add, x, y))) == p_cm and \
                             not (p_cm == 0 and (tuple(x) == tuple(y))), \
                                                it.product(lookup_p3, repeat=2))
    lookup_si = it.ifilter(lambda x : _abs2(x) == p_cm, lookup_p3)
    lookup_p = it.ifilter(lambda ((w,x),y): \
                            tuple(it.imap(operator.add, w, x)) == tuple(it.imap(operator.neg, y)), \
                                               it.product(lookup_so, lookup_si))

  elif diagram.startswith('C4'):
    lookup_so = it.ifilter(lambda (x,y): \
#                             _abs2(list(it.imap(operator.add, x, y))) == p_cm and \
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

# TODO: currently calculates all combinations, but only g1 with g01 etc. is wanted,
# not eg. g1 with g02
# TODO: gamma_5 is hardcoded. That should be generelized in the future
# TODO: combine set_lookup_p and set_lookup_g (and set_lookup_d) to possible set_lookup_qn?
def set_lookup_g(gammas, diagram):
  """
  create lookup table for all combinations of Dirac operators that can appear 
  in the gevp of a chosen operator basis

  Parameters
  ----------
  gammas : list of list of ints and string
      A list which for each gamma structure coupling to the rho meson contains
      a list with the integer indices used in the contraction code and a 
      latex-style name for plotting labels.
  diagram : string, {'C20', 'C2+', 'C3+', 'C4*'}
      If one- and two-meson operators contribute, the appearing Dirac operators
      change because the mesons in a two-meson operator can couple to different
      quantum numbers individually

  Returns
  -------
  list of tuples (of tuples)
      List that contains tuples for source and sink with tuples for the Dirac
      operators. They are referenced by their id in the cntrv0.1-code. For
      every possible operators combination, there is one list entry
  """

  if diagram == 'C20':
    lookup_so = it.product([g for gamma in gammas for g in gamma[:-1]])
    lookup_so, lookup_si = it.tee(lookup_so, 2)
  elif diagram == 'C2+':
    lookup_so = it.product([5])
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
  diagram : string, {'C20', 'C2+', 'C3+', 'C4+B', 'C4+D'}
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

  The quantum numbers are momenta and gamma structures at source and sink time.
  Depending on the number of quark lines there can be tuples of quantum numbers
  at the same lattice site
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

# TODO: displacement hardcoded
def set_groupname(diagram, p, g):
  """
  Creates string with filename of the desired diagram calculated by the 
  cntrv0.1-code.

  Parameters
  ----------
  diagram : string, {'C20', 'C2+', 'C3+', 'C4+B', 'C4+D'}
      Diagram of wick contractions for the rho meson.
  p : tuple (of tuple) of tuple of int
      Tuple of momentum (or tuple of tuple of momenta in case of two-meson 
      operators) at source and sink. Momenta are given as tuple of int. 
  g : tuple (of tuple) of int
      Tuple of Dirac operators (or tuple of tuple of Dirac operators) at source
      and sink. Dirac operators are referred toby their id in the cntrv0.1-code.
      For every possible operators combination, there is one list entry.

  Returns
  -------
  groupname : string
      Filename of contracted perambulators for the given parameters
  """
  if diagram.startswith('C2'):
    groupname = diagram + '_uu_p%1i%1i%1i.d000.g%i' % \
                                             (p[0][0], p[0][1], p[0][2], g[0][0]) \
                    + '_p%1i%1i%1i.d000.g%i' % (p[1][0], p[1][1], p[1][2], g[1][0])
  elif diagram.startswith('C3'):
    groupname = diagram + '_uuu_p%1i%1i%1i.d000.g5' % \
                                                   (p[0][0][0], p[0][0][1], p[0][0][2]) \
                        + '_p%1i%1i%1i.d000.g%1i' % \
                                                (p[1][0], p[1][1], p[1][2], g[1][0]) \
                        + '_p%1i%1i%1i.d000.g5' % (p[0][1][0], p[0][1][1], p[0][1][2])
  elif diagram == 'C4+D' or diagram == 'C4+C':
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

def multiply_trtr_diagram(data):
  """
  Multiply factors of the tr()*tr() like diagram C4+D

  Parameters
  ----------
  data : pd.DataFrame
      A dataframe with rows cnfg x T x {rere, reim, imre, imim}. 

  Returns
  -------
  data : pd.DataFrame
      A dataframe with rows cnfg x T x {re, im} appropriately multiplied out.

  Notes
  -----
  C4+D is the diagram factorizing into a product of two traces. The imaginary 
  part of the individual traces is 0 in the isosping limit. To suppress noise, 
  The real part of C4+D are calculated as product of real parts of the single 
  traces:
  (a+ib)*(c+id) ~= (ac) +i(bc+ad)
  because b, d are noise
  """

  names = data.index.names
  data = pd.concat([ data.xs('rere', level=2), \
                     data.xs('reim', level=2) + data.xs('imre', level=2)], \
                   keys=['re', 'im']).reorder_levels([1,2,0]).\
                                                         sort_index(level=[0,1])
  data.index.names = names
  return data

################################################################################
# reading configurations

def read(lookup_cnfg, lookup_qn, diagram, T, directory, verbose=0):
  """
  Read resulting correlators from contraction code and creates a pd.DataFrame

  Parameters
  ----------
  lookup_cnfg : list of int
      List of the gauge configurations to read
  lookup_qn : pd.DataFrame
      pd.DataFrame with every row being a set of physical quantum numbers to be 
      read
  diagram : string, {'C20', 'C2+', 'C3+', 'C4+B', 'C4+D'}
      Diagram of wick contractions for the rho meson.
  T : int
      Time extent of the lattice
  directory : string
      Output path of contraction code

  Returns
  -------
  data : pd.DataFrame
      A pd.DataFrame with rows (cnfg x T x re/im) and columns i where i are 
      the row numbers of `lookup_qn` 
  """

  data = []
  comb = True if diagram == 'C4+D' else False

  for cnfg in lookup_cnfg:
    # filename and path
    filename = directory + '/' + diagram + '_cnfg%i' % cnfg + '.h5'
    try:
      fh = h5py.File(filename, "r")
    except IOError:
      print 'file %s not found' % filename
      raise

    # to achieve hirarchical indexing for quantum numbers build DataFrame for
    # each loop seperately
    # TODO: is it necessary to build that completely or can that be 
    # constructed by successively storing each operator with pd.HDFStore()?
    data_qn = DataFrame()
#    print DataFrame(lookup_p)
#    print DataFrame(lookup_g)

    for op in lookup_qn.index:
      p = lookup_qn.ix[op, ['p_{so}', 'p_{si}']]
      g = lookup_qn.ix[op, ['\gamma_{so}', '\gamma_{si}']]
      groupname = set_groupname(diagram, p, g)

      # read data from file as numpy array and interpret as complex
      # numbers for easier treatment
      try:
        tmp = np.asarray(fh[groupname]).view(complex)
      except ValueError:
        print("could not read %s for config %d" % (groupname, cnfg))
        continue

      # in case diagram is C4+D perform last mutliplication of factorizing
      # traces
      # the file contains 4 numbers per time slice: ReRe, ReIm, ImRe, and ImIm,
      # here combined 2 complex number
      if comb:
        # reshaping so we can extract the data easier
        tmp = tmp.reshape((-1,2))
        # extracting right combination, assuming ImIm contains only noise
        dtmp = 1.j * (tmp[:,1].real + tmp[:,0].imag) + tmp[:,0].real
        tmp = dtmp.copy()

      # save data into data frame
      data_qn[op] = pd.DataFrame(tmp, columns=['re/im'])
    data.append(data_qn)
  data = pd.concat(data, keys=lookup_cnfg, axis=0, names=['cnfg', 'T'])

  if verbose:
    print '\tfinished reading'

  return data.sort_index(level=[0,1])

def read_old(lookup_cnfg, lookup_qn, diagram, T, directory, verbose=0):
  """
  Read resulting correlators from contraction code and creates a pd.DataFrame

  Parameters
  ----------
  lookup_cnfg : list of int
      List of the gauge configurations to read
  lookup_qn : pd.DataFrame
      pd.DataFrame with every row being a set of physical quantum numbers to be 
      read
  diagram : string, {'C20', 'C2+', 'C3+', 'C4+B', 'C4+D', 'C4+C'}
      Diagram of wick contractions for the rho meson.
  T : int
      Time extent of the lattice
  directory : string
      Output path of contraction code

  Returns
  -------
  data : pd.DataFrame
      A pd.DataFrame with rows (cnfg x T x re/im) and columns i where i are 
      the row numbers of `lookup_qn` 
  """

  data = []

  for cnfg in lookup_cnfg:
    # filename and path
    filename = directory + '/' + diagram + '_cnfg%i' % cnfg + '.h5'
    try:
      fh = h5py.File(filename, "r")
    except IOError:
      print 'file %s not found' % filename
      raise

    # to achieve hirarchical indexing for quantum numbers build DataFrame for
    # each loop seperately
    # TODO: is it necessary to build that completely or can that be 
    # constructed by successively storing each operator with pd.HDFStore()?
    data_qn = pd.DataFrame()
#    print DataFrame(lookup_p)
#    print DataFrame(lookup_g)

    for op in lookup_qn.index:
      p = lookup_qn.ix[op, ['p_{so}', 'p_{si}']]
      g = lookup_qn.ix[op, ['\gamma_{so}', '\gamma_{si}']]
      # TODO: catch when a groupname does not exist
      groupname = set_groupname(diagram, p, g)

      # TODO: real and imaginay part are treated seperately through the whole
      # program. It might be easiert to combine them already at read-in or 
      # even better after wick contraction because of different structure for
      # C4+D
      try:
        tmp = np.asarray(fh[groupname])
      except ValueError:
        print("could not read %s for config %d" % (groupname, cnfg))
        continue

      print(tmp[0])
      print(tmp.shape)
      print(tmp.dtype)
      data_qn[op] = pd.DataFrame(tmp, columns=['re/im'])

    data.append(data_qn)
    fh.close()
  data = pd.concat(data, keys=lookup_cnfg, axis=0, names=['cnfg'])
  #print(data)

  if verbose:
    print '\tfinished reading'

  return data.sort_index(level=[0,1])
  ##############################################################################


