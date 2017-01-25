#!/hiskp2/werner/libraries/Python-2.7.12/python
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

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

##################################################################################

p_cm = 0
def build_gevp(p_cm, irrep, verbose):

  correlator = 'C2'
  path = './readdata/%s_p%1i_subduced_and_averaged.h5' % (correlator, p_cm)
  subduced_2pt = utils.read_hdf5_correlators(path, read_qn=False)
  
  correlator = 'C3'
  path = './readdata/%s_p%1i_subduced_and_averaged.h5' % (correlator, p_cm)
  subduced_3pt = utils.read_hdf5_correlators(path, read_qn=False)
  
  correlator = 'C4'
  path = './readdata/%s_p%1i_subduced_and_averaged.h5' % (correlator, p_cm)
  subduced_4pt = utils.read_hdf5_correlators(path, read_qn=False)

  left = pd.concat([subduced_2pt, subduced_3pt]).reset_index()
  subduced_3pt_T = subduced_3pt.swaplevel(0,1)
  subduced_3pt_T.index.set_names(['gevp_row', 'gevp_col'], inplace=True)
  right = pd.concat([subduced_3pt_T, subduced_4pt]).reset_index()

  gevp = pd.merge(left, right, how='outer').\
                                set_index(['gevp_row', 'gevp_col']).sort_index() 

  ################################################################################
  # writing data to file
  # TODO: catch if len(gevp.index) is not a square number. Can that happen?
  gevp_size = np.sqrt(len(gevp.index))
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
        np.asarray(gevp.ix[0].values).reshape(gevp.ix[0].unstack().shape), path, \
        verbose)

  return gevp


