# TODO: Implement logic to automatically create filenames and folders e.g. with regex and os.walk

import itertools as it
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

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

def compare_files(paths, keys):
  """
  Read two pd.DataFrames from hdf5 files and compare whether the content is the same

  Parameters
  ----------
  paths : list of string
      Paths to the hdf5 files
  keys : list of string
      The hdf5 groupnames to access the given data under

  Returns
  -------
  bool
      True if Data is equal

  Raises
  ------
  ValueError
      If the DataFrames are differently labeled
  """

  dfs = []
  for path, key in zip(paths, keys):
    dfs.append(read_hdf5_correlators(path, key))

  return np.all(dfs[0] == dfs[1])

paths = ['A40.24-for-testing', 'A40.24-for-testing_cmp']
#folders = ['0_raw-data', '1_subduced-data', '2_contracted-data', '3_gevp-data']
folders = ['3_gevp-data']

diagrams = ['C20', 'C3+', 'C4+B', 'C4+D']
correlators = ['C2', 'C3', 'C4']
gevp = ['Gevp']
momenta = range(4)

#for i in it.product(paths, folders_diagrams):
#  print ''.join(i)
  
if len(paths) != 2:
  print 'ERROR: Use with two paths to compare'
  exit(0)
print 'Checking equality of ' + paths[0] + ' and ' + paths[1], '\n'

for folder in folders:
  if folder.startswith('0') or folder.startswith('1'):
    nametags = diagrams
  elif folder.startswith('2'):
    nametags = correlators
  elif folder.startswith('3'):
    nametags = gevp

  for name in nametags:
    checks = []
    for p in momenta: 

      if p == 0:
        irreps = ['T1u']
      elif p == 1:
        irreps = ['A1g', 'Ep1g']
      elif p == 2:
        irreps = ['A1g', 'A2g', 'A2u']
      elif p == 3:
        irreps = ['A1g', 'Ep1g']

      for irrep in irreps:
        path1 = '%s/%s/' % (paths[0], folder)
        path2 = '%s/%s/' % (paths[1], folder)
        filename = '%s_p%1d' % (name, p)
        if not folder.startswith('0'):
          filename = filename + '_%s' % (irrep)
        filename = filename + '.h5'

  
        try:
          checks.append(compare_files([path1+filename,path2+filename], ['data','data']))
        except ValueError:
          checks.append(False)
        print '\tChecking %20s' % (filename) + ':\t', checks[-1]
    print folder + '/' + name + ': ', np.all(np.array(checks))







