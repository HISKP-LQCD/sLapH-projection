import os

import pandas as pd
from pandas import Series, DataFrame

################################################################################
# checks if the directory where the file will be written does exist
def ensure_dir(f):
#  d = os.path.dirname(f)
  if not os.path.exists(f):
    os.makedirs(f)

def read_hdf5_correlators(path):
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
  data.columns.name = 'index'
  qn = pd.read_hdf(path, 'qn')

  return data, qn

  
def write_hdf5_correlators(path, data, lookup_qn):
  """
  write pd.DataFrame of correlation functions as hdf5 file

  Parameters
  ----------
  path : string
      Path to store the hdf5 file
  data : pd.DataFrame
      The correlator data contained in the hdf5 file
  lookup_qn : pd.DataFrame
      The physical quantum number associated to the data above
  """

  store = pd.HDFStore(path)
  # write all operators
  store['data'] = data
  store['qn'] = lookup_qn

  store.close()
  
  print '\tfinished writing'

