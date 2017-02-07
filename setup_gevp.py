#!/hiskp2/werner/libraries/Python-2.7.12/python
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import utils

##################################################################################

# TODO: factor out the setup of subduced_npt and just pass a 
# list of lists of pd.DataFrame
def build_gevp(data, irrep, verbose):
  """
  Create a single pd.DataFrame containing all correlators contributing to the 
  rho gevp.

  Parameters
  ----------
  data : Dictionary of pd.DataFrame, keys in ({'C2', 'C3', 'C4'}, `irrep`)

      For each correlator `data` must contain an associated pd.DataFrame with 
      its completely summed out subduced and contracted lattice data

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
  subduced_2pt = data[(correlator, irrep)]
  
  correlator = 'C3'
  subduced_3pt = data[(correlator, irrep)]
  
  subduced_3pt_T = subduced_3pt.swaplevel(0,1)
  subduced_3pt_T.index.set_names(['gevp_row', 'gevp_col'], inplace=True)

  correlator = 'C4'
  subduced_4pt = data[(correlator, irrep)]

  ############################################################################## 

  # gevp = C2   C3
  #        C3^T C4
  upper = pd.concat([subduced_2pt, subduced_3pt])
  lower = pd.concat([subduced_3pt_T, subduced_4pt])
  gevp = pd.concat([upper, lower]).sort_index()

  return gevp




