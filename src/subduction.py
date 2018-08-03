#!/usr/bin/python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import utils


def set_lookup_corr(coefficients_irrep, qn, verbose=1):
    """
    Calculate table with all required coefficients and quantum numbers of the
    states needed to obtain eigenstates under a certain irreducible representation

    Parameters
    ----------
    coefficients_irrep: pd.DataFrame
        Table with all quantum numbers on source and sink as columns and the rows
        of the irreducible representations as rows. Also contains two columns
        with the appropriate Clebsch Gordan coefficient for all these quantum
        numbers

    qn : pd.DataFrame
        pd.DataFrame with every row being a set of physical quantum numbers
        contributing to `diagram`. The rows are indexed by a unique identifier

    Returns
    -------
    lookup_corr: pd.DataFrame
        Table with a column for each quantum number at source and sink, the
        Clebsch-Gordan coefficients at source and sink and a column decoding
        the row and column of the gevp these quantum numbers enter into:
        \mu \gamma_{so} coefficient_{so} p_{so} \gamma_{si} \
            coefficient_{si} p_{si} index gevp_row gevp_col
    """

    # associate clebsch-gordan coefficients with the correct qn index
    # TODO: Check whether left merge results in nan somewhere as in this case
    # data is missing
    lookup_corr = pd.merge(coefficients_irrep.reset_index(), qn.reset_index(),
                           how='left')

    # Add two additional columns with the same string if the quantum numbers
    # describe equivalent physical constellations: gevp_row and gevp_col
    if('q_{so}' in lookup_corr.columns):
        q_string = ', q: ' + lookup_corr['q_{so}']
    else:
        q_string = ''

    lookup_corr['gevp_row'] = \
        'p: ' + lookup_corr['p_{cm}'].apply(eval).apply(utils._abs2).apply(str) + \
        q_string + ', g: ' + lookup_corr['operator_label_{so}']

    if('q_{si}' in lookup_corr.columns):
        q_string = ', q: ' + lookup_corr['q_{si}']
    else:
        q_string = ''

    lookup_corr['gevp_col'] = \
        'p: ' + lookup_corr['p_{cm}'].apply(eval).apply(utils._abs2).apply(str) + \
        q_string + ', g: ' + lookup_corr['operator_label_{si}']

    lookup_corr.drop(['operator_label_{so}',
                      'q_{so}',
                      'q_{si}',
                      'operator_label_{si}'],
                     axis=1,
                     inplace=True,
                     errors='ignore')

    # Set index as it shall appear in projected correlators
    index = lookup_corr.columns.difference(['index', 'coefficient']).tolist()
    order = {r'Irrep': 0,
             r'mult': 1,
             r'gevp_row': 2,
             r'gevp_col': 3,
             r'p_{cm}': 4,
             r'\mu': 5,
             r'\beta': 6,
             r'p^{0}_{so}': 7,
             r'p^{1}_{so}': 8,
             r'p^{0}_{si}': 9,
             r'p^{1}_{si}': 10,
             r'\gamma^{0}_{so}': 11,
             r'\gamma^{1}_{so}': 12,
             r'\gamma^{0}_{si}': 13,
             r'\gamma^{1}_{si}': 14}
    index = sorted(index, key=lambda x: order[x])
    lookup_corr.set_index(index, inplace=True)

    if verbose >= 1:
        print 'lookup_corr'
    if verbose == 1:
        print lookup_corr.head()
    if verbose >= 2:
        print lookup_corr

    return lookup_corr


def project_correlators(data, qn_irrep):
    """
    Combine physical operators to transform like a given irreducible
    representation and combine equivalent physical quantum numbers

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned und munged raw output of the cntr-v.0.1 code
    qn_irrep : pd.Series
        Series with a column for each quantum number at source and sink, the
        Clebsch-Gordan coefficients at source and sink and a column decoding
        the row and column of the gevp these quantum numbers enter into:
        \mu \gamma_{so} coefficient_{so} p_{so} \gamma_{si} \
            coefficient_{si} p_{si} index gevp_row gevp_col

    Returns
    -------
    subduced : pd.DataFrame
        Table with physical quantum numbers as rows and gauge configuration and
        lattice time as columns.
        Contains the linear combinations of correlation functions transforming
        like what the parameters qn_irrep was created with
    """

    # actual subduction step. sum conj(cg_so) * cg_si * corr
    projected_correlators = pd.merge(qn_irrep, data.T,
                                     how='left', left_on=['index'], right_index=True)
    del projected_correlators['index']

    projected_correlators = projected_correlators[
        projected_correlators.columns.difference(['coefficient'])].\
        multiply(projected_correlators['coefficient'], axis=0)

    projected_correlators.columns = pd.MultiIndex.from_tuples(projected_correlators.columns,
                                                              names=('cnfg', 'T'))

    projected_correlators = projected_correlators.sort_index()

    # I do not know why the dtype got converted to object, but convert it back
    # to complex
    return projected_correlators.apply(pd.to_numeric).sort_index()
