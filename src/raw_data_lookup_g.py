import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# TODO: gamma_5 is hardcoded. That should be generelized in the future
# TODO: Refactor set_lookup with different lists for scalar,  vector, etc.
# TODO: Is there a more elegent method to get a product than merging
#       Dataframes with index set to 0? itertools? :D
def set_lookup_g(gamma_labels, diagram):
    """
    create lookup table for all combinations of Dirac operators that can appear
    in the gevp of a chosen operator basis

    Parameters
    ----------
    gamma_labels : list of string
        A list which for each gamma structure coupling to the rho meson contains
        a list with the integer indices used in the contraction code and a
        latex-style name for plotting labels.
    diagram : string, {'C20', 'C2c', 'C3c', 'C4*'}
        If one- and two-meson operators contribute, the appearing Dirac operators
        change because the mesons in a two-meson operator can couple to different
        quantum numbers individually

    Returns
    -------
    list of tuples (of tuples)
        List that contains tuples for source and sink with tuples for the Dirac
        operators. They are referenced by their id in the cntrv0.1-code. For
        every possible operators combination, there is one list entry
    """

    gamma_dic = {'gamma_5': DataFrame({'\gamma': [5]}),
                 'gamma_0': DataFrame({'\gamma': [0]}),
                 'gamma_i': DataFrame({'\gamma': [1, 2, 3]}),
                 'gamma_50i': DataFrame({'\gamma': [13, 14, 15]})
                 }

    if diagram == 'C20':
        J_so = 1
        J_si = 1

        # TODO: Thats kind of an ugly way to obtain a flat list of all gamma ids
        #       used in sLapH-contractions for the given gamma_labels
        gamma_so = pd.concat([gamma_dic[gl_so] for gl_so in gamma_labels[J_so]])
        gamma_so = gamma_so.rename(columns={'\gamma': '\gamma^{0}'})

        gamma_si = gamma_so

    elif diagram == 'C2c':

        J_so = 0
        J_si = 0

        gamma_so = pd.concat([gamma_dic[gl_so] for gl_so in gamma_labels[J_so]])
        gamma_so = gamma_so.rename(columns={'\gamma': '\gamma^{0}_{so}'})

        gamma_si = pd.concat([gamma_dic[gl_si] for gl_si in gamma_labels[J_si]])
        gamma_si = gamma_si.rename(columns={'\gamma': '\gamma^{0}_{si}'})

    elif diagram == 'C3c':

        J_so = 0
        J_si = 1

        gamma_so = pd.concat([gamma_dic[gl_so] for gl_so in gamma_labels[J_so]])

        gamma_so['tmp'] = 0
        gamma_so = pd.merge(gamma_so, gamma_so,
                            how='outer',
                            on=['tmp'],
                            suffixes=['^{0}', '^{1}'])
        gamma_so = gamma_so.rename(columns={'\gamma^{0}': '\gamma^{0}_{so}',
                                            '\gamma^{1}': '\gamma^{1}_{so}'})
        del(gamma_so['tmp'])

        gamma_si = pd.concat([gamma_dic[gl_si] for gl_si in gamma_labels[J_si]])
        gamma_si = gamma_si.rename(columns={'\gamma': '\gamma^{0}_{si}'})

    elif diagram.startswith('C4'):

        J_so = 0
        J_si = 0

        gamma_so = pd.concat([gamma_dic[gl_so] for gl_so in gamma_labels[J_so]])
        gamma_so['tmp'] = 0
        gamma_so = pd.merge(gamma_so, gamma_so,
                            how='outer',
                            on=['tmp'],
                            suffixes=['^{0}', '^{1}'])
        gamma_so = gamma_so.rename(columns={'\gamma^{0}': '\gamma^{0}_{so}',
                                            '\gamma^{1}': '\gamma^{1}_{so}'})
        del(gamma_so['tmp'])

        gamma_si = pd.concat([gamma_dic[gl_si] for gl_si in gamma_labels[J_si]])
        gamma_si['tmp'] = 0
        gamma_si = pd.merge(gamma_si, gamma_si,
                            how='outer',
                            on=['tmp'],
                            suffixes=['^{0}', '^{1}'])
        gamma_si = gamma_si.rename(columns={'\gamma^{0}': '\gamma^{0}_{si}',
                                            '\gamma^{1}': '\gamma^{1}_{si}'})
        del(gamma_si['tmp'])

    else:
        print 'in set_lookup_g: diagram unknown! Quantum numbers corrupted.'
        return

    gamma_so['tmp'] = 0
    gamma_si['tmp'] = 0
    lookup_g = pd.merge(gamma_so, gamma_si,
                        how='outer',
                        on=['tmp'],
                        suffixes=['_{so}', '_{si}'])

    del(lookup_g['tmp'])

    return lookup_g
