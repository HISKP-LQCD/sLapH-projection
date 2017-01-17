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
  Helper function to read correlators in the temporary format of the subduction
  code

  Parameters
  ---------
  path : string
    Path to the hdf5 file

  Returns
  -------

  pd.DataFrame
    The correlator data contained in the hdf5 file
  pd.DataFrame
    The physical quantum number associated to the data above
  """

  data = pd.read_hdf(path, 'data')
  data.columns.name = 'index'
  qn = pd.read_hdf(path, 'qn')

  return data, qn

