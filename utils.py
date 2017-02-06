# Helper functions for IO to store intermediate results of subduction code on
# hard disk
# TODO: could be restructured as table format to access individual files and 
# allow appending, but speed is uncritical
import os

import pandas as pd
from pandas import Series, DataFrame

################################################################################
# checks if the directory where the file will be written does exist
def ensure_dir(f):
  """Helper function to create a directory if it does not exis"""
  if not os.path.exists(f):
    os.makedirs(f)

def read_hdf5_correlators(path, key):
  """
  Read pd.DataFrame from hdf5 file

  Parameters
  ----------
  path : string
      Path to the hdf5 file
  key : string
      The hdf5 groupname to access the given data under. 

  Returns
  -------
  data : pd.DataFrame
      The data contained in the hdf5 file under the given key
  """

  data = pd.read_hdf(path, key)
  
  return data
  
def write_hdf5_correlators(path, filename, data, key, verbose=False):
  """
  write pd.DataFrame as hdf5 file

  Parameters
  ----------
  path : string
      Path to store the hdf5 file
  filename : string
      Name to save the hdf5 file as
  data : pd.DataFrame
      The data to write
  key : string
      The hdf5 groupname to access the given data under. Specifying multiple 
      keys, different data can be written to the same file
  """

  ensure_dir(path)
  data.to_hdf(path+filename, key, mode='w')
 
  if verbose:
    print '\tfinished writing', filename

