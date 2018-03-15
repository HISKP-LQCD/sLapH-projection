import itertools as it
import numpy as np

import pandas as pd
from pandas import Series, DataFrame

from ast import literal_eval

from utils import _abs2

def set_lookup_p_for_one_particle(lookup_p3, p_cm):
  """
  Create lookup table for all 3-momenta p_0, used for a single particle

  Parameters
  ----------
  lookup_p3 : pd.DataFrame
      List of all 3-vectors a single particle could have
  p_cm : int
      Absolute value of sum of all momenta at source or sink. Must be equal at
      both due to momentum conservation

  Returns
  -------
  pd.DataFrame
      DataFrame that contains one column for the particle and a row for every 
      possible combination of momenta respecting momentum conservation and the 
      cutoff given. Also contains one column for the total momentum to merge on.
  """

  # Restrict set of 3-momenta to those with the correct abulute value
  # Set multiindex to allow combination with two-particle operators
  lookup = DataFrame.copy(lookup_p3[lookup_p3['p'].apply(_abs2) == p_cm])
  lookup.columns = ['p^{0}']

  # Total momentum is equal to the particle's momentum
  lookup['p_{cm}'] = lookup['p^{0}']

  return lookup.applymap(str)

def set_lookup_p_for_two_particles(lookup_p3, p_max, p_cm, skip=False):
  """
  Create lookup table for all possible 3-momenta (p_0, p_1), that two particles 
  can have

  Parameters
  ----------
  lookup_p3 : pd.DataFrame
      List of all 3-vectors a single particle could have
  p_max : int
      Cut-off. Do not consider combinations where |p_0|^2+|p_1|^2 <= p_max
  p_cm : int
      Absolute value of sum of all momenta at source or sink. Must be equal at
      both due to momentum conservation
  skip : bool, optional
      In the rho analysis the pions cannot be at rest, so skip these momentum
      combinations.

  Returns
  -------
  pd.DataFrame
      DataFrame that contains a column for each particle and a row for every 
      possible combination of momenta respecting momentum conservation and the 
      cutoff given.
  """

  # DataFrame with all combinations of 3-momenta
  lookup = pd.merge(lookup_p3, lookup_p3, how='outer', \
                       left_index=True, right_index=True)
  lookup.columns = ['p^{0}', 'p^{1}']

  # Total momentum is equal to the sum of the particle's momenta
  lookup['p_{cm}'] = map(lambda k1, k2: tuple([sum(x) for x in zip(k1,k2)]), \
                                lookup['p^{0}'], lookup['p^{1}'])

  # Restrict set of 3-momenta to those with the correct abulute value
  lookup = lookup[lookup['p_{cm}'].apply(_abs2) == p_cm]
  # Restrict set of 3-momenta to those where |p1|+|p2| <= p_max
  lookup = lookup[lookup['p^{0}'].apply(_abs2) + lookup['p^{1}'].apply(_abs2) <= p_max]

  # For rho analysis, explicitely exclude S-wave:
  # \pi(0,0,0) + \pi(0,0,0) -> \rho(0,0,0) forbidden by angular momentum 
  # conservation
  if skip:
    lookup = lookup[(lookup['p^{0}'] != (0,0,0)) | (lookup['p^{1}'] != (0,0,0))]

  return lookup.applymap(str)

def set_lookup_p(p_max, p_cm, diagram, skip=False):
  """
  create lookup table for all possible 3-momenta that can appear on the lattice
  below a given cutoff

  Parameters
  ----------
  p_max : int
      Cut-off. Do not consider momenta with a higher absolute value than this
  p_cm : int
      Absolute value of sum of all momenta at source or sink. Must be equal at
      both due to momentum conservation
  diagram : string, {'C20', 'C2+', 'C3+', 'C4*'}
      The number of momenta is equal to the number of quarks in a chosen 
      diagram. It is encoded in the second char of the diagram name
  skip : bool, optional
      In the rho analysis the pions cannot be at rest, so skip these momentum
      combinations.

  Returns
  -------
  pd.DataFrame
      DataFrame that contains a (hierarchical) column for each particle 
      involved in the diagram and a row for every possible combination of 
      momenta respecting momentum conservation and the cutoff given by p_max.
  """

  # for moving frames, sum of individual component's absolute value (i.e.
  # total kindetic energy) might be larger than center of mass absolute 
  # value. Modify the cutoff accordingly.
  if p_cm == 0:
    p_max = 4
  elif p_cm == 1:
    p_max = 5
  elif p_cm == 2:
    p_max = 6
  elif p_cm == 3:
    p_max = 7
  elif p_cm == 4:
    p_max = 4


  # List of all 3-vectors a single particle could have
  p_max_for_one_particle = 4
  max_possible_component = np.ceil(np.sqrt(p_max_for_one_particle)).astype(int)
  lookup_p3 = list(it.ifilter(lambda x: _abs2(x) <= p_max_for_one_particle, \
                              it.product(range(-max_possible_component, 
                                          max_possible_component+1), repeat=3)))
  lookup_p3 = pd.DataFrame(zip(lookup_p3), columns=['p']) 
  lookup_p3.index = np.repeat(0, len(lookup_p3))

  if diagram.startswith('C2'):
    lookup_so = set_lookup_p_for_one_particle(lookup_p3, p_cm)
    lookup_si = set_lookup_p_for_one_particle(lookup_p3, p_cm)

  elif diagram.startswith('C3'):
    lookup_so = set_lookup_p_for_two_particles(lookup_p3, p_max, p_cm, skip)
    lookup_si = set_lookup_p_for_one_particle(lookup_p3, p_cm)

  elif diagram.startswith('C4'):
    lookup_so = set_lookup_p_for_two_particles(lookup_p3, p_max, p_cm, skip)
    lookup_si = set_lookup_p_for_two_particles(lookup_p3, p_max, p_cm, skip)

  else:
    print 'in set_lookup_p: diagram unknown! Quantum numbers corrupted.'
 
  # DataFrame with all combinations of source and sink with same total momentum
  lookup_p = pd.merge(lookup_so, lookup_si, on=['p_{cm}'], \
                      suffixes=['_{so}', '_{si}'])

  return lookup_p


