import itertools as it
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import utils


def rho_2pt(data, verbose=1):
    """
    Perform Wick contraction for 2pt function

    Parameters
    ----------
    data : Dictionary of pd.DataFrame, keys in {'C20', 'C3c', 'C4cB', 'C4cD')

        For each diagram constituting the given `corrrelator` `data` must contain
        an associated pd.DataFrame with its subduced lattice data

    Returns
    -------
    wick : pd.DataFrame
        pd.DataFrame with indices like `data['C20']` and Wick contractions
        performed.

    Notes
    -----
    The rho 2pt function is given by the contraction.
    C^\text{2pt} = \langle \rho(t_{so})^\dagger \rho(t_si) \rangle

    The gamma strucures that can appear in \rho(t) are hardcoded
    """

    gamma_i = [1, 2, 3]
    gamma_50i = [13, 14, 15]

    assert {'C20'} <= set(data.keys()), 'Subduced data must contain C20'

    wick = data['C20']

    # Warning: 1j hardcoded
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_i) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_i)] *= (1.)
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_i) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_50i)] *= (1j)
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_50i) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_i)] *= -(1j)
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_50i) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_50i)] *= (1.)

    if verbose >= 2:
        print "wick['C2']"
    if verbose == 2:
        print wick.head()
    if verbose == 3:
        print wick

    return wick


def pipi_2pt(data, verbose=1):
    """
    Perform Wick contraction for 2pt function

    Parameters
    ----------
    data : Dictionary of pd.DataFrame, keys in ('C20', 'C3c', 'C4cB', 'C4cD')

        For each diagram constituting the given `corrrelator` `data` must contain
        an associated pd.DataFrame with its subduced lattice data

    Returns
    -------
    wick : pd.DataFrame
        pd.DataFrame with indices like `data['C20']` and Wick contractions
        performed.

    Notes
    -----
    The gamma strucures that can appear in \pi\pi(t) are hardcoded
    """
    wick = data['C2c']

    return wick


def pi_2pt(data, verbose=1):
    """
    Perform Wick contraction for 2pt function

    Parameters
    ----------
    data : Dictionary of pd.DataFrame, keys in ('C20', 'C3c', 'C4cB', 'C4cD')

        For each diagram constituting the given `corrrelator` `data` must contain
        an associated pd.DataFrame with its subduced lattice data

    Returns
    -------
    wick : pd.DataFrame
        pd.DataFrame with indices like `data['C20']` and Wick contractions
        performed.

    Notes
    -----
    The gamma strucures that can appear in \pi\pi(t) are hardcoded
    """

    gamma_5 = [5]
    gamma_0 = [0]

    wick = data['C2c']

#    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0)] *= (1j)
#    wick[wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_0)] *= (1j)

    return wick


################################################################################

def rho_3pt(data, verbose=1):
    """
    Perform Wick contraction for 3pt function

    Parameters
    ----------
    data : Dictionary of pd.DataFrame, keys in ('C20', 'C3c', 'C4cB', 'C4cD')

        For each diagram constituting the given `corrrelator` `data` must contain
        an associated pd.DataFrame with its subduced lattice data

    Returns
    -------
    wick : pd.DataFrame

        pd.DataFrame with indices like `data['C30']` and Wick contractions
        performed.

    Notes
    -----
    The rho 3pt function is given by the contraction.
    C^\text{3pt} = \langle \pi\pi(t_{so})^\dagger \rho(t_si) \rangle

    The gamma strucures that can appear in \rho(t) are hardcoded
    """

    gamma_0 = [0]
    gamma_5 = [5]
    gamma_i = [1, 2, 3]
    gamma_50i = [13, 14, 15]

    assert {'C3c'} <= set(data.keys()), 'Subduced data must contain C3c'

    wick = data['C3c']

    # This is only the lower left part of the Gevp. For the upper right 3pt part the
    # coefficient have to be complex conjugated and one needs to calculate <rho pipi>
    # which will amount to the same result.
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_5) &
         wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_5) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_i)] *=  np.sqrt(2.) * (1j)
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_5) &
         wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_i)] *= -np.sqrt(2.)
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0) &
         wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_5) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_i)] *= -np.sqrt(2.)
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0) &
         wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_i)] *=  np.sqrt(2.) * (1j)

    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_5) &
         wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_5) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_50i)] *= -np.sqrt(2.)
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_5) &
         wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_50i)] *= -np.sqrt(2.) * (1j)
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0) &
         wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_5) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_50i)] *= -np.sqrt(2.) * (1j)
    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0) &
         wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0) &
         wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_50i)] *= -np.sqrt(2.)

    if verbose >= 2:
        print "wick['C3']"
    if verbose == 2:
        print wick.head()
    if verbose == 3:
        print wick

    return wick

##########################################################################################

# TODO: catch if keys were not found


def rho_4pt(data, verbose=0):
    """
    Perform Wick contraction for 4pt function

    Parameters
    ----------
    data : Dictionary of pd.DataFrame, keys in ('C20', 'C3c', 'C4cB', 'C4cD')

        For each diagram constituting the given `corrrelator` `data` must contain
        an associated pd.DataFrame with its subduced lattice data

    Returns
    -------
    wick : pd.DataFrame

        pd.DataFrame with indices like the union of indices in `data['C4cB']` and
        cdata['C4cD']` and Wick contractions performed.

    Notes
    -----
    The rho 3pt function is given by the contraction.
    C^\text{2pt} = \langle \pi\pi(t_{so})^\dagger \pi\pi(t_si) \rangle

    The gamma strucures that can appear in \rho(t) are hardcoded
    """

    gamma_0 = [0]
    gamma_5 = [5]

    assert {'C4cD', 'C4cB'} <= set(
        data.keys()), 'Subduced data must contain C4cD and C4cB'

    data_box = data['C4cB']

#    data_box_sign = (data_box.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0)
#                    + data_box.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0)
#                    + data_box.index.get_level_values('\gamma^{0}_{si}').isin(gamma_0)
#                    + data_box.index.get_level_values('\gamma^{1}_{si}').isin(gamma_0) 
#                    + 1)%2 + \
#                    (data_box.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0)
#                    + data_box.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0)
#                    + data_box.index.get_level_values('\gamma^{0}_{si}').isin(gamma_0)
#                    + data_box.index.get_level_values('\gamma^{1}_{si}').isin(gamma_0))%2 * (1j)
#    data_box_sign *= -8
#
#    data_box = data_box.multiply(data_box_sign, axis=0)

    data_box_sign = ((data_box.reset_index()[['\gamma^{0}_{so}', '\gamma^{1}_{so}', '\gamma^{0}_{si}', '\gamma^{1}_{si}']].isin(gamma_0).sum(axis=1) % 2) != 0).values
    data_box[data_box_sign] *= 1j

    data_box *= -2

    # TODO: support read in if the passed data is incomplete
#  data_box = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[0], p_cm), 'data')
#  data_dia = pd.read_hdf('readdata/%s_p%1i.h5' % (diagrams[1], p_cm), 'data')

    data_dia = data['C4cD']

#    data_dia_sign = (+ data_dia.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0)
#                    + data_dia.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0) 
#                    + data_dia.index.get_level_values('\gamma^{0}_{si}').isin(gamma_0)
#                    + data_dia.index.get_level_values('\gamma^{1}_{si}').isin(gamma_0)
#                    + 1)%2 + \
#                    (data_dia.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0)
#                    + data_dia.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0)
#                    + data_dia.index.get_level_values('\gamma^{0}_{si}').isin(gamma_0)
#                    + data_dia.index.get_level_values('\gamma^{1}_{si}').isin(gamma_0))%2 *(1j)

    # Bug in contractions: C4cD built for pi^+ pi^+. Flipping back by complex conjugation
    # of one term. This changes to sign of the complete term if g^0 must be commuted with
    # g^5
#    data_dia_sign = (+ data_dia.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0)
#                    + data_dia.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0) 
#                    + 1)%2 + \
#                    (data_dia.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0)
#                    + data_dia.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0))%2 *(1j)
#
#    data_dia_sign *= (data_dia.index.get_level_values('\gamma^{0}_{si}').isin(gamma_0)
#                    + data_dia.index.get_level_values('\gamma^{1}_{si}').isin(gamma_0) 
#                    + 1)%2 + \
#                    (data_dia.index.get_level_values('\gamma^{0}_{si}').isin(gamma_0)
#                    + data_dia.index.get_level_values('\gamma^{1}_{si}').isin(gamma_0))%2 * (1j)

#    data_dia_sign *= 4


#    data_dia = data_dia.multiply(data_dia_sign, axis=0)

#    data_dia[data_dia.index.get_level_values('\gamma^{0}_{so}').isin(gamma_5)
#             & data_dia.index.get_level_values('\gamma^{0}_{si}').isin(gamma_5)
#             & data_dia.index.get_level_values('\gamma^{1}_{so}').isin(gamma_5)
#             & data_dia.index.get_level_values('\gamma^{1}_{si}').isin(gamma_5)] *= 1j
#

    data_dia_sign = ((data_dia.reset_index()[['\gamma^{0}_{so}', '\gamma^{1}_{so}', '\gamma^{0}_{si}', '\gamma^{1}_{si}']].isin(gamma_0).sum(axis=1) % 2) != 0).values
    data_dia[data_dia_sign] *= 1j

#    data_dia_sign = ((data_dia.reset_index()[['\gamma^{0}_{so}', '\gamma^{0}_{si}']].isin(gamma_0).sum(axis=1) % 2) != 0).values
#    data_dia[data_dia_sign] *= 1j
#    data_dia_sign = ((data_dia.reset_index()[['\gamma^{0}_{so}', '\gamma^{0}_{si}']].isin(gamma_0).sum(axis=1) % 2) != 0).values
#    data_dia[data_dia_sign] *= 1j

    wick = data_dia.add(data_box, fill_value=0)

#    wick[wick.index.get_level_values('\gamma^{0}_{so}').isin(gamma_0) &
#             wick.index.get_level_values('\gamma^{1}_{so}').isin(gamma_0)] *= 1j
#    wick[wick.index.get_level_values('\gamma^{0}_{si}').isin(gamma_0) &
#             wick.index.get_level_values('\gamma^{1}_{si}').isin(gamma_0)] *= 1j


    if verbose >= 2:
        print "wick['C4']"
    if verbose == 2:
        print wick.head()
    if verbose == 3:
        print wick

    return wick


def pipi_4pt(data, verbose=0):
    """
    Perform Wick contraction for 4pt function

    Parameters
    ----------
    data : Dictionary of pd.DataFrame, keys in ('C20', 'C3c', 'C4cB', 'C4cD')

        For each diagram constituting the given `corrrelator` `data` must contain
        an associated pd.DataFrame with its subduced lattice data

    Returns
    -------
    wick : pd.DataFrame

        pd.DataFrame with indices like the union of indices in `data['C4cB']` and
        `data['C4cD']` and Wick contractions performed.

    Notes
    -----
    The rho 3pt function is given by the contraction.
    C^\text{2pt} = \langle \pi\pi(t_{so})^\dagger \pi\pi(t_si) \rangle

    The gamma strucures that can appear in \rho(t) are hardcoded
    """

    data_cro = data['C4cC']
    data_dia = data['C4cD']

    wick = ((-1.) * data_cro).add(data_dia, fill_value=0)

    return wick

################################################################################


def set_lookup_correlators(diagrams):
    """
    Extracts the correlation functions one can construct from the given diagrams

    Parameters
    ----------
    diagrams : list of string {'C20', 'C3c', 'C4cB', 'C4cD'}
        Diagram of wick contractions contributing the rho meson correlation
        function

    Returns
    -------
    lookup_correlators : list of string {'C2', 'C3', 'C4'}
        Correlation functions constituted by given diagrams
    """

    lookup_correlators = {}
    for nb_quarklines in range(2, 5):

        # as a function argument give the names of all diagrams with the correct
        # number of quarklines
        mask = [d.startswith('C%1d' % nb_quarklines) for d in diagrams]
        diagram = list(it.compress(diagrams, mask))

        if len(diagram) != 0:
            lookup_correlators.update({"C%1d" % nb_quarklines: diagram})

    return lookup_correlators


def contract_correlators(process, data, correlator, verbose=0):
    """
    Sums all diagrams with the factors they appear in the Wick contractions

    Parameters
    ----------
    data : Dictionary of pd.DataFrame, keys in {'C20', 'C3c', 'C4cB', 'C4cD'}
        For each diagram constituting the given `corrrelator` `data` must contain
        an associated pd.DataFrame with its subduced lattice data
    correlator : string {'C2', 'C3', 'C4'}
        Correlation functions to perform the Wick contraction on

    Returns
    -------
    contracted : pd.DataFrame
        A correlation function with completely performed Wick contractions. Rows
        and columns are unchanged compared to `data`

    Notes
    -----
    The correlation functions contributing to the rho gevp can be characterized
    by the number of quarklines
    2pt \langle \rho(t_{so})^\dagger \rho(t_si) \rangle
    3pt \langle \pi\pi(t_{so})^\dagger \rho(t_si) \rangle
    4pt \langle \pi\pi(t_{so})^\dagger \pi\pi(t_si) \rangle
    In the isospin limit the first 2 only have one (linearly independent) diagram
    contributing, while the last one has two.
    """

    # TODO: I don't think you have to emulate c function pointers for this
    rho = {'C2': rho_2pt, 'C3': rho_3pt, 'C4': rho_4pt}
    pipi = {'C2': pipi_2pt, 'C4': pipi_4pt}
    pi = {'C2': pi_2pt}

    if process == 'rho':
        # rho analysis
        contracted = rho[correlator](data, verbose)
    elif process == 'pipi':
        # pipi I=2 analysis
        contracted = pipi[correlator](data, verbose)
    elif process == 'pi':
        # pi analysis
        contracted = pi[correlator](data, verbose)

    return contracted
