import os

import pandas as pd
from pandas import Series, DataFrame

################################################################################
# checks if the directory where the file will be written does exist
def ensure_dir(f):
#  d = os.path.dirname(f)
  if not os.path.exists(f):
    os.makedirs(f)

def read_hdf5_correlators(path, read_qn=True):
  """
  Read correlators in the format used by subduction code routines

  Parameters
  ----------
  path : string
      Path to the hdf5 file

  Returns
  -------
  data : pd.DataFrame
      The correlator data contained in the hdf5 file
  qn : pd.DataFrame
      The physical quantum number associated to the data above
  """

  data = pd.read_hdf(path, 'data')
  if read_qn:
    data.index.name = 'id'
    qn = pd.read_hdf(path, 'qn')

    return data, qn
  else:
    return data
  
# TODO: us pandas to_hdf5 routines
def write_hdf5_correlators(path, filename, data, lookup_qn, verbose=False):
  """
  write pd.DataFrame of correlation functions as hdf5 file

  Parameters
  ----------
  path : string
      Path to store the hdf5 file
  filename : string
      Name to save the hdf5 file as
  data : pd.DataFrame
      The correlator data contained in the hdf5 file
  lookup_qn : pd.DataFrame
      The physical quantum number associated to the data above
  """

  ensure_dir(path)
  store = pd.HDFStore(path+filename)
  # write all operators
  store['data'] = data
  if lookup_qn is not None:
    store['qn'] = lookup_qn

  store.close()
  
  if verbose:
    print '\tfinished writing', filename

