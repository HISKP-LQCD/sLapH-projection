# functions for munging and cleaning correlators

import h5py
import numpy as np
import itertools as it

from utils import _minus
from raw_data_lookup_p import set_lookup_p

import functools

import pandas as pd
from pandas import Series, DataFrame

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

# TODO: gamma_5 is hardcoded. That should be generelized in the future
# TODO: combine set_lookup_p and set_lookup_g (and set_lookup_d) to possible set_lookup_qn?
# TODO: Refactor set_lookup with different lists for scalar,  vector, etc.
# TODO: Is there a more elegent method to get a product than merging 
#       Dataframes with index set to 0? itertools? :D
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

    lookup_so = DataFrame([g for gamma in gammas for g in gamma[:-1]])
    lookup_so.index = np.repeat(0, len(lookup_so))
    lookup_so.columns = pd.MultiIndex.from_tuples( [('\gamma', 0)] )

    lookup_si = lookup_so

  elif diagram == 'C2+':

    lookup_so = DataFrame([5])
    lookup_so.index = np.repeat(0, len(lookup_so))
    lookup_so.columns = pd.MultiIndex.from_tuples( [('\gamma', 0)] )

    lookup_si = lookup_so

  elif diagram == 'C3+':

    lookup = DataFrame([5])
    lookup.index = np.repeat(0, len(lookup))

    lookup_so = pd.merge(lookup, lookup, left_index=True, right_index=True)
    lookup_so.columns = pd.MultiIndex.from_tuples( [('\gamma', 0), ('\gamma', 1) ] )

    lookup_si = DataFrame([g for gamma in gammas for g in gamma[:-1]])
    lookup_si.index = np.repeat(0, len(lookup_si))
    lookup_si.columns = pd.MultiIndex.from_tuples( [('\gamma', 0)] )

  elif diagram.startswith('C4'):

    lookup = DataFrame([5])
    lookup.index = np.repeat(0, len(lookup))

    lookup_so = pd.merge(lookup, lookup, left_index=True, right_index=True)
    lookup_so.columns = pd.MultiIndex.from_tuples( [('\gamma', 0), ('\gamma', 1) ] )

    lookup_si = lookup_so

  else:
    print 'in set_lookup_g: diagram unknown! Quantum numbers corrupted.'
    return

  lookup_g = pd.merge(lookup_so, lookup_si, left_index=True, right_index=True,
                      suffixes=['_{so}', '_{si}'])

  print diagram
  print lookup_g

  return lookup_g

def set_lookup_qn(diagram, p_cm, p_max, gammas, process='pipi', verbose=0):
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

  skip = True if process == 'rho' else False

  lookup_p = set_lookup_p(p_max, p_cm, diagram, skip)
  lookup_p.index = np.repeat(0, len(lookup_p))
  lookup_g = set_lookup_g(gammas, diagram)
  lookup_g.index = np.repeat(0, len(lookup_g))

  lookup_qn = pd.merge(lookup_p, lookup_g, how='left', left_index=True, right_index=True)
  lookup_qn.reset_index(drop=True, inplace=True)
  
  print lookup_qn
  return lookup_qn

def set_groupname(diagram, s):
  """
  Creates string with filename of the desired diagram calculated by the 
  cntrv0.1-code.

  Parameters
  ----------
  diagram : string, {'C20', 'C2+', 'C3+', 'C4+B', 'C4+D'}
      Diagram of wick contractions for the rho meson.
  s : pd.Series
      Contains momenta and gamma structure the groupnames shall be built with.
      Index is given by Multiindex where the first level is 'p_{so}', 'p_{si}', 
      '\gamma_{so}' or '\gamma_{si}' and the second level id of the particle
      Dirac operators are referred to by their id in the sLapH-contractions
      code. 

  Returns
  -------
  groupname : string
      Filename of contracted perambulators for the given parameters

  Notes
  -----
  The filename contains a convential minus sign for the sink momenta!

  Function takes a series as argument in order to be called in DataFrame.apply()
  """

  p_so = s['p_{so}']
  g_so = s['\gamma_{so}']

  p_si = s['p_{si}'].apply(_minus)
  g_si = s['\gamma_{si}']

  if diagram.startswith('C2'):
    groupname = diagram \
                  + '_uu_p%1i%1i%1i.d000.g%i' % ( p_so[0] + (g_so[0],) ) \
                  +    '_p%1i%1i%1i.d000.g%i' % ( p_si[0] + (g_si[0],) ) 
  elif diagram.startswith('C3'):
    groupname = diagram \
                  + '_uuu_p%1i%1i%1i.d000.g%1i' % ( p_so[0] + (g_so[0],) ) \
                  +     '_p%1i%1i%1i.d000.g%1i' % ( p_si[0] + (g_si[0],) ) \
                  +     '_p%1i%1i%1i.d000.g%1i' % ( p_so[1] + (g_so[1],) )
  elif diagram == 'C4+D' or diagram == 'C4+C':
    groupname = diagram \
                  + '_uuuu_p%1i%1i%1i.d000.g%1i' % ( p_so[0] + (g_so[0],) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_si[0] + (g_si[0],) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_so[1] + (g_so[1],) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_si[1] + (g_si[1],) )
  elif diagram == 'C4+B':
    groupname = diagram \
                  + '_uuuu_p%1i%1i%1i.d000.g%1i' % ( p_so[0] + (g_so[0],) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_si[0] + (g_si[0],) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_si[1] + (g_si[1],) ) \
                  +      '_p%1i%1i%1i.d000.g%1i' % ( p_so[1] + (g_so[1],) ) 
  else:
    print 'in set_groupname: diagram unknown! Quantum numbers corrupted.'
    return

  return groupname

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

  comb = True if diagram == 'C4+D' else False

  groupname = lookup_qn.apply(functools.partial(set_groupname, diagram), axis=1)

  data = []

  for cnfg in lookup_cnfg:
    # filename and path
    filename = directory + '/' + diagram + '_cnfg%04i' % cnfg + '.h5'
    try:
      fh = h5py.File(filename, "r")
    except IOError:
      print 'file %s not found' % filename
      raise

    data_fh = DataFrame()

    for op in lookup_qn.index:

      # read data from file as numpy array and interpret as complex
      # numbers for easier treatment
      try:
        tmp = np.asarray(fh[groupname[op]]).view(complex)
      except KeyError:
        print("could not read %s for config %d" % (groupname, cnfg))
        continue
  
      # C4+D is the diagram factorizing into a product of two traces. The imaginary 
      # part of the individual traces is 0 in the isosping limit. To suppress noise, 
      # The real part of C4+D are calculated as product of real parts of the single 
      # traces:
      # (a+ib)*(c+id) ~= (ac) +i(bc+ad)
      # because b, d are noise
      if comb:
        # reshaping so we can extract the data easier
        tmp = tmp.reshape((-1,2))
        # extracting right combination, assuming ImIm contains only noise
        dtmp = 1.j * (tmp[:,1].real + tmp[:,0].imag) + tmp[:,0].real
        tmp = dtmp.copy()
  
      # save data into data frame
      data_fh[op] = pd.Series(tmp)
    data.append(data_fh)
  data = pd.concat(data, keys=lookup_cnfg, axis=0, names=['cnfg', 'T'])
  
 
  data.sort_index(level=[0,1], inplace=True)

  if verbose >= 2:
    print lookup_qn
  if verbose:
    print '\tfinished reading\n'
  if verbose >= 3:
    print data.mean(axis=1).apply(np.real)

  return data

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
    # open file
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
    ndata = 0
    nfailed = 0

    for op in lookup_qn.index:
      ndata += 1
      # generate operator name
      p = lookup_qn.ix[op, ['p_{so}', 'p_{si}']]
      g = lookup_qn.ix[op, ['\gamma_{so}', '\gamma_{si}']]
      groupname = set_groupname(diagram, p, g)

      # read operator from file and store in data frame
      try:
        tmp = np.asarray(fh[groupname])
      except KeyError:
        #if diagram == 'C4+C' and cnfg == 714:
        #  print("could not read %s for config %d" % (groupname, cnfg))
        nfailed += 1
        continue
      data_qn[op] = pd.DataFrame(tmp, columns=['re/im'])
    if nfailed > 0 and verbose > 0:
      print("could not read %d of %d data" % (nfailed, ndata))

    # append all data for one config and close the file
    data.append(data_qn)
    fh.close()
  # generate data frame containing all operators for all configs
  data = pd.concat(data, keys=lookup_cnfg, axis=0, names=['cnfg', 'T'])
  data.sort_index(level=[0,1], inplace=True)

  if verbose >= 2:
    print lookup_qn
  if verbose:
    print '\tfinished reading\n'
  if verbose >= 3:
    print data.mean(axis=1).apply(np.real)

  return data
  ##############################################################################


