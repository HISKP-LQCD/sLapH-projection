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


def read_hdf5_correlators(path, key='key'):
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

    data = pd.read_hdf(path, key)

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


def write_data_ascii(data, filename, verbose=0):
    """
    Writes the data into a file.

    Parameters
    ----------
    data: np.array
        A 2d numpy array with data. shape = (nsamples, T)
    filename: string
        The filename of the file.

    Notes
    -----
    Taken from Christians analysis-code https://github.com/chjost/analysis-code

    The file is written to have L. Liu's data format so that the first line
    has information about the number of samples and the length of each sample.
    """
    if verbose:
        print("Saving to file " + str(filename))

    # in case the dimension is 1, treat the data as one sample
    # to make the rest easier we add an extra axis
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    # init variables
    nsamples = data.shape[0]
    T = data.shape[1]
    L = int(T / 2)
    # write header
    head = "%i %i %i %i %i" % (nsamples, T, 0, L, 0)
    # prepare data and counter
    #_data = data.flatten()
    _data = data.reshape((T * nsamples), -1)
    _counter = np.fromfunction(lambda i, *j: i % T,
                               (_data.shape[0],) + (1,) * (len(_data.shape) - 1), dtype=int)
    _fdata = np.concatenate((_counter, _data), axis=1)
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


def write_ascii_correlators(filename, data, verbose=1):
    """
    write pd.DataFrame as ascii file in Liuming's format

    Parameters
    ----------
    path : string
        Path to store the hdf5 file
    filename : string
        Name to save the hdf5 file as
    data : pd.DataFrame
        The data to write
    """

    path = os.path.dirname(filename)
    ensure_dir(path)
    write_data_ascii(np.asarray(pd_series_to_np_array(data)), filename, verbose)


def write_ascii_gevp(path, basename, data, verbose=1):

    data = data.T.reset_index().set_index(['cnfg', 'T']).stack(level=['gevp_row', 'gevp_col', 'p_{cm}', '\mu']).reset_index(['p_{cm}', '\mu', 'cnfg', 'T'])

    assert np.all(data.notnull()), ('Gevp contains null entires')
    assert gmpy.is_square(len(data.index.unique())), 'Gevp is not a square matrix'

    data_size = gmpy.sqrt(len(data.index.unique()))

    if verbose:
        print 'Creating a %d x %d Gevp' % (data_size, data_size)

    ensure_dir(path)
    f = open(os.path.join(path, basename + '_indices.txt'), 'w')
    f.write("%8s\tphysical content\n" % "Element")

    for counter in range(len(data.index)):

        filename = os.path.join(path, basename + '.%d.%d.dat' % (counter / data_size, counter % data_size))

        # Write file with physical content corresponding to index number (gevp_col)
        if counter < data_size:
            f.write("%12d\t%s\n" % (counter, data.index[counter][1]))

        # TODO: with to_csv this becomes a onliner but Liumings head format will
        # be annoying. Also the loop can probably run over data.iterrows()
        write_ascii_correlators(filename, data.ix[counter], verbose)

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
