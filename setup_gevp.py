#!/hiskp2/werner/libraries/Python-2.7.12/python
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import gmpy

import utils

################################################################################
# TODO: write that for a pandas dataframe with hierarchical index nb_cnfg x T
def write_data_ascii(data, filename, verbose=False):
  """
  Writes the data into a file.
  
  Parameters
  ----------
  filename: string
      The filename of the file.
  data: np.array
      A 2d numpy array with data. shape = (nsamples, T)

  Notes
  -----
  Taken from Christians analysis-code https://github.com/chjost/analysis-code

  The file is written to have L. Liu's data format so that the first line
  has information about the number of samples and the length of each sample.
  """
  if verbose:
    print("saving to file " + str(filename))
  
  # in case the dimension is 1, treat the data as one sample
  # to make the rest easier we add an extra axis
  if len(data.shape) == 1:
    data = data.reshape(1, -1)
  # init variables
  nsamples = data.shape[0]
  T = data.shape[1]
  L = int(T/2)
  # write header
  head = "%i %i %i %i %i" % (nsamples, T, 0, L, 0)
  # prepare data and counter
  #_data = data.flatten()
  _data = data.reshape((T*nsamples), -1)
  _counter = np.fromfunction(lambda i, *j: i%T,
                             (_data.shape[0],) + (1,)*(len(_data.shape)-1), dtype=int)
  _fdata = np.concatenate((_counter,_data), axis=1)
  # generate format string
  fmt = ('%.0f',) + ('%.14f',) * _data[0].size
  # write data to file
  np.savetxt(filename, _fdata, header=head, comments='', fmt=fmt)

def pd_series_to_np_array(series):
  """
  Converts a pandas Series to a numpy array

  Parameters
  ----------
  series : pd.Series
      Written for data with 1 column and a 2-level hierarchical index
  
  Returns
  -------
  np.array
      In the case of a Series with 2-level hierarchical index this will be a 
      2d array with the level 0 index as rows and the level 1 index as columns
  """

  return np.asarray(series.values).reshape(series.unstack().shape)

##################################################################################

p_cm = 0
# TODO: factor out the setup of subduced_npt and just pass a 
# list of lists of pd.DataFrame
def build_gevp(p_cm, irrep, verbose):
  """
  Create a single pd.DataFrame containing all correlators contributing to the 
  rho gevp.

  Parameters
  ----------
  p_cm : int, {0,1,2,3,4}
      Center of mass momentum of the lattice. Used to specify the appropriate
      little group of rotational symmetry
  irrep : string, {'T1', 'A1', 'E2', 'B1', 'B2'}
      name of the irreducible representation of the little group all operators
      of the gevp are required to transform under.

  Returns
  -------
  gevp : pd.DataFrame
      Table with a row for each gevp element (sorted by gevp column running
      faster than gevp row) and hierarchical columns for gauge configuration 
      number and timeslice
  """

  ############################################################################## 
  # read and prepare correlation function to enter the gevp
  # TODO: 2x2 kinds of operators hardcoded. E.g. for baryons this has to be
  # adopted.
  correlator = 'C2'
  path = './readdata/%s_p%1i_%s_avg.h5' % (correlator, p_cm, irrep)
  subduced_2pt = utils.read_hdf5_correlators(path, read_qn=False)
  
  correlator = 'C3'
  path = './readdata/%s_p%1i_%s_avg.h5' % (correlator, p_cm, irrep)
  subduced_3pt = utils.read_hdf5_correlators(path, read_qn=False)
  
  subduced_3pt_T = subduced_3pt.swaplevel(0,1)
  subduced_3pt_T.index.set_names(['gevp_row', 'gevp_col'], inplace=True)

  correlator = 'C4'
  path = './readdata/%s_p%1i_%s_avg.h5' % (correlator, p_cm, irrep)
  subduced_4pt = utils.read_hdf5_correlators(path, read_qn=False)

  ############################################################################## 

  # gevp = C2   C3
  #        C3^T C4

  # reset index to merge only corresponding gevp entries
  upper = pd.concat([subduced_2pt, subduced_3pt]).reset_index()
  lower = pd.concat([subduced_3pt_T, subduced_4pt]).reset_index()

  # merge on gevp entries and configuration number, timeslice. Reset gevp_entry
  # as index afterwards
  # TODO merge or is a simple concat(axis=0) also enought?
  gevp = pd.merge(upper, lower, how='outer').\
                                set_index(['gevp_row', 'gevp_col']).sort_index() 

  assert np.all(gevp.notnull()), 'Gevp contains null entires'
  assert gmpy.is_square(len(gevp.index)), 'Gevp is not a square matrix'
  ##############################################################################
  # writing data to file
  gevp_size = gmpy.sqrt(len(gevp.index))
  if verbose:
    print 'creating a %d x %d Gevp' % (gevp_size, gevp_size)

  for counter in range(len(gevp.index)):
    # TODO: put that into utils
    path = 'gevp/p%1i/%s/Rho_Gevp_p%1d_%s.%d.%d.dat' % (p_cm, irrep, \
                  p_cm, irrep, counter/gevp_size, counter%gevp_size)
    utils.ensure_dir('./gevp')
    utils.ensure_dir('./gevp/p%1i' % p_cm)
    utils.ensure_dir('./gevp/p%1i/%s' % (p_cm, irrep))
    # TODO: with to_csv this becomes a onliner but Liumings head format will 
    # be annoying. Also the loop can probably run over gevp.iterrows()
    write_data_ascii( \
             np.asarray(pd_series_to_np_array(gevp.ix[counter])), path, verbose)

  path = './readdata/Gevp_p%1i_%s.h5' % (p_cm, irrep)
  utils.ensure_dir('./readdata')
  utils.write_hdf5_correlators(path, gevp, None)


  return gevp


