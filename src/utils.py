from ast import literal_eval
import gmpy
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import errno
import itertools as it
import operator
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

################################################################################
# Convenience function to work with three-momenta in pd.DataFrames


def _scalar_mul(x, y):
    return sum(it.imap(operator.mul, x, y))


def _abs2(x):
    return _scalar_mul(x, x)


def _minus(x):
    # Python distinguishes +0 and -0. I explicitly want the + for string output
    return tuple(-np.array(x) + 0)

################################################################################
# checks if the directory where the file will be written does exist
# See https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python


def ensure_dir(f):
    """Helper function to create a directory if it does not exist"""
#  if not os.path.exists(f):
    try:
        os.makedirs(f)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(f):
            pass
        else:
            raise

################################################################################
# IO routines


def read_hdf5_correlators(filename, key='key'):
    """
    Read pd.DataFrame from hdf5 file

    Parameters
    ----------
    path : string
        Path to the hdf5 file
    key : string
        The hdf5 groupname to access the given data under. Defaults to something very
        imaginative

    Returns
    -------
    data : pd.DataFrame
        The data contained in the hdf5 file under the given key
    """

    data = pd.read_hdf(filename, key)

    return data


def write_hdf5_correlators(filename, data, verbose=0, key='key'):
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
        The hdf5 groupname to access the given data under. Defaults to something very
        imaginative
    """

    path = os.path.dirname(filename)
    ensure_dir(path)
    data.to_hdf(filename, key, mode='w')

    if verbose >= 1:
        print '\tFinished writing', filename

################################################################################
# TODO: write that for a pandas dataframe with hierarchical index nb_cnfg x T


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


# TODO: The loop structure is really hacky because I essentially want two multindices
# to loop over
def write_ascii_gevp(path, basename, data, verbose=1):

    ensure_dir(path)

    # Cast data into longish data format with only gevp_row and gevp_col as index
    data = data.T.reset_index().set_index(['cnfg', 'T']).stack(
        level=['gevp_row', 'gevp_col', 'p_{cm}', '\mu']).reset_index(['p_{cm}', '\mu', 'cnfg', 'T'])
    data[['p_x', 'p_y', 'p_z']] = data['p_{cm}'].apply(literal_eval).apply(pd.Series)
    del data['p_{cm}']
    data.rename(columns={0: 'value', '\mu': 'alpha'}, inplace=True)

    gevp_indices = data.index.unique()
    operator_indices = data.loc[gevp_indices[0]].set_index(
        ['p_x', 'p_y', 'p_z', 'alpha']).index.unique()

    assert np.all(data.notnull()), ('Gevp contains null entires')
    assert gmpy.is_square(len(gevp_indices)), 'Gevp is not a square matrix'

    gevp_size = gmpy.sqrt(len(gevp_indices))

    # Write file with physical content corresponding to gevp index number (gevp_col)
    gevp_elements = [i[1] for i in gevp_indices.values[:gevp_size]]
    np.savetxt(os.path.join(path,
                            basename + '_gevp-indices.tsv'),
               np.array(zip(range(gevp_size),
                            gevp_elements)),
               fmt='%s',
               delimiter='\t',
               header='id\telement')

    # Write file with physical content corresponding to operator (p_cm, alpha)
    operator_elements = np.array([tuple(i) for i in operator_indices.values])
    np.savetxt(
        os.path.join(path, basename + '_operator-indices.tsv'),
        np.column_stack(
            (np.arange(
                operator_elements.shape[0]),
                operator_elements)),
        fmt='%s',
        delimiter='\t',
        header='id\tp_x\tp_y\tp_z\talpha')

    if verbose:
        print 'Creating a {} x {} Gevp from {} operators'.format(
            gevp_size, gevp_size, operator_elements.shape[0])

    # Loop over all projected operators and gevp elements and write an ascii file
    for gevp_counter, gevp_index in enumerate(gevp_indices):
        for operator_counter, operator_index in enumerate(operator_indices):

            filename = os.path.join(
                path,
                basename +
                '_op%d_gevp%d.%d.tsv' %
                (operator_counter,
                 gevp_counter /
                 gevp_size,
                 gevp_counter %
                 gevp_size))

            df = data.loc[gevp_index].set_index(
                ['p_x', 'p_y', 'p_z', 'alpha']).loc[operator_index]

            df.to_csv(filename, sep='\t', index=False)

################################################################################
# Convenience function to create pdf files


def create_pdfplot(filename):
    """
    Helper function to create a pdfplot object and ensure existence of the path
    """

    path = os.path.dirname(filename)
    ensure_dir(path)
    pdfplot = PdfPages(filename)

    return pdfplot
