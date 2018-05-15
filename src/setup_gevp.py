#!/hiskp2/werner/libraries/Python-2.7.12/python
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import utils

##################################################################################

# TODO: factor out the setup of subduced_npt and just pass a
# list of lists of pd.DataFrame
def build_gevp(data, mode, verbose):
    """
    Create a single pd.DataFrame containing all correlators contributing to the
    rho gevp.

    Parameters
    ----------
    data : Dictionary of pd.DataFrame, keys in ('C2', 'C3', 'C4')

        For each correlator `data` must contain an associated pd.DataFrame with
        its completely summed out subduced and contracted lattice data

    mode : string, {'rho', 'pipi'}

        Unique identifier for identification of identity of analysis

    target_irrep : string, {'A1u', 'A2u', 'E1u', 'Ep1u', 'T1u', 'T2u', 'G1u', 'G2u',
                     'K1u', 'K2u', 'A1g', 'A2g', 'E1g', 'Ep1g', 'T1g', 'T2g',
                     'G1g', 'G2g', 'K1g', 'K2g'}

        name of the irreducible representation of the little group all operators
        of the gevp are required to transform under.

    Returns
    -------
    gevp : pd.DataFrame

        Table with a row for each gevp element (sorted by gevp column running
        faster than gevp row) and hierarchical columns for gauge configuration
        number and timeslice
    """

    if mode == 'pipi':

        print "Warning: I=2 Gevp currently not implemented"
        gevp = data["C4"]
        # NOTE: Delete all rows and colums with only NaN's
        gevp = gevp.dropna(axis=0,how='all').dropna(axis=1,how='all')

    elif mode == 'rho':

        assert set(data.keys()) == {'C2', 'C3', 'C4'}, 'Gevp must contain C2, C3 and C4'

        ##############################################################################
        # read and prepare correlation function to enter the gevp
        # TODO: 2x2 kinds of operators hardcoded. E.g. for baryons this has to be
        # adopted.
        correlator = 'C2'
        subduced_2pt = data[correlator]

        correlator = 'C3'
        subduced_3pt = data[correlator]

        subduced_3pt_T = subduced_3pt.swaplevel('gevp_row','gevp_col')

        index_3pt_T = map({'gevp_row' : 'gevp_col', 'gevp_col' : 'gevp_row'}.get, 
                subduced_3pt_T.index.names, subduced_3pt_T.index.names)
        subduced_3pt_T.index.set_names(index_3pt_T, inplace=True)

        correlator = 'C4'
        subduced_4pt = data[correlator]


        ##############################################################################

        # gevp = C2   C3
        #        C3^T C4
        upper = pd.concat([subduced_2pt, subduced_3pt_T])
        lower = pd.concat([subduced_3pt, subduced_4pt])
        gevp = pd.concat([upper, lower]).sort_index()

        if verbose >= 1:
            print 'gevp'
        if verbose == 1:
            print gevp.head()
        if verbose >= 2:
            print gevp

    return gevp
