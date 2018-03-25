import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# TODO: gamma_5 is hardcoded. That should be generelized in the future
# TODO: combine set_lookup_p and set_lookup_g (and set_lookup_d) to possible set_lookup_qn?
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
  diagram : string, {'C20', 'C2+', 'C3+', 'C4*'}
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

  gamma_dic = { 'gamma_5' :   DataFrame({'\gamma' : [5]}),
                'gamma_05':   DataFrame({'\gamma' : [6]}),
                'gamma_i' :   DataFrame({'\gamma' : [1,2,3]}),
                'gamma_50i' : DataFrame({'\gamma' : [13,14,15]})
              }

  if diagram == 'C20':
    J_so = 1
    J_si = 1

    # TODO: Thats kind of an ugly way to obtain a flat list of all gamma ids 
    #       used in sLapH-contractions for the given gamma_labels
    gamma_so = pd.concat([gamma_dic[gl_so] for gl_so in gamma_labels[J_so]])
    gamma_so = gamma_so.rename(columns={'\gamma' : '\gamma^{0}'})

    gamma_si = gamma_so 

  elif diagram == 'C2+':

    J_so = 0
    J_si = 0

    gamma_so = pd.concat([gamma_dic[gl_so] for gl_so in gamma_labels[J_so]])
    gamma_so = gamma_so.rename(columns={'\gamma' : '\gamma^{0}_{so}'})

    gamma_si = pd.concat([gamma_dic[gl_si] for gl_si in gamma_labels[J_si]])
    gamma_si = gamma_si.rename(columns={'\gamma' : '\gamma^{0}_{si}'})


  elif diagram == 'C3+':

    J_so = 0
    J_si = 1

    gamma_so = pd.concat([gamma_dic[gl_so] for gl_so in gamma_labels[J_so]])
    gamma_so = gamma_so.rename(columns={'\gamma' : '\gamma^{0}_{so}'})
    gamma_so['\gamma^{1}_{so}'] = gamma_so['\gamma^{0}_{so}']

    gamma_si = pd.concat([gamma_dic[gl_si] for gl_si in gamma_labels[J_si]])
    gamma_si = gamma_si.rename(columns={'\gamma' : '\gamma^{0}_{si}'})

  elif diagram.startswith('C4'):

    lookup = DataFrame([5])
    lookup.index = np.repeat(0, len(lookup))

    lookup_so = pd.merge(lookup, lookup, left_index=True, right_index=True)
    lookup_so.columns = ['\gamma^{0}', '\gamma^{1}']

    lookup_si = lookup_so

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

  print diagram
  print lookup_g

  return lookup_g

