import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import itertools as it
import cmath
import functools
import os
import operator
import collections

from asteval import Interpreter
aeval = Interpreter()
aeval.symtable['I'] = 1j

from ast import literal_eval

from utils import _scalar_mul, _abs2, _minus

# TODO: path for groups is hardcoded here. Shift that into clebsch-gordan module
def return_cg(p_cm, irrep):
  """
  Creates table with eigenstates of an irreducible representation created from
  a Clebsch-Gordan decomposition of two pseudoscalar particles with momenta 
  k_1 and k_2

  Parameters
  ----------

    p_cm : int, {0,1,2,3,4}
        Center of mass momentum of the lattice. Used to specify the appropriate
        little group of rotational symmetry. Absolute value of an integer 
        3-vector
    irrep : string
        Specifying the irreducible representation operators should transform 
        under

  Returns
  -------

    pd.DataFrame
        Table with cg-coefficients for each momentum combination p=(k_1, k_2)
        and row \mu
        Has columns J, M, cg-coefficient, p, \mu and unnamed indices

  Notes
  -----

    J, M are both hardcoded to (0,0) referring to scattering of two 
    (pseudo)scalars

  See
  ---

    clebsch_gordan.example_cg
  """

  prefs = [[0.,0.,0.], [0.,0.,1.], [0.,1.,1.], [1.,1.,1.], [0.,0.,2.]]
#           [0.,1.,2.], [1.,1.,2.]]
  p2max = len(prefs)

  # initialize groups
  S = 1./np.sqrt(2.)
  # tells clebsch_gordan to use cartesian basis
  U3 = np.asarray([[0,0,-1.],[1.j,0,0],[0,1,0]])
  U2 = np.asarray([[S,S],[1.j*S,-1.j*S]])

  path = os.path.normpath(os.path.join(os.getcwd(), "groups/"))
  groups = group.init_groups(prefs=prefs, p2max=p2max, U2=U2, U3=U3,
          path=path)

  # define the particles to combine
  j1 = 0 # J quantum number of particle 1
  j2 = 0 # J quantum number of particle 2
  ir1 = [ g.subduction_SU2(int(j1*2+1)) for g in groups]
  ir2 = [ g.subduction_SU2(int(j2*2+1)) for g in groups]

  # calc coefficients
  df = DataFrame()
  for (i, i1), (j, i2) in it.product(zip(range(p2max), ir1), zip(range(p2max), ir2)):
    for _i1, _i2 in it.product(i1, i2):
      try:

        cgs = group.TOhCG(p_cm, i, j, groups, ir1=_i1, ir2=_i2)

        # TODO: irreps explizit angeben. TOh gibt Liste der beitragenden irreps 
        # zurueck TOh.subduction_SU2(j) mit j = 2j+1
        #cgs = group.TOhCG(0, p, p, groups, ir1="A2g", ir2="T2g")
        #print("pandas")
        df = pd.concat([df, cgs.to_pandas()], ignore_index=True)
      except RuntimeError:
        continue

  df.rename(columns={'row' : '\mu', 'multi' : 'mult', 
                                       'cg' : 'cg-coefficient'}, inplace=True)
  df['cg-coefficient'] = df['cg-coefficient'].apply(aeval)


  # Create new column 'p' with tuple of momenta 
  # ( (p1x, p1y, p1z), (p2x, p2y, p2z) )
  # TODO warning for imaginary parts
  def to_tuple(list, sign=+1):
      return tuple([int(sign*l.real) for l in list])
  df['p1'] = df['p1'].apply(to_tuple)
  df['p2'] = df['p2'].apply(to_tuple)
  df['p'] = list(zip(df['p1'], df['p2']))
  df.drop(['p1', 'p2', 'ptot'], axis=1, inplace=True)

  # Inserting J, M to merge with basis and obtain gamma structure
  # Hardcoded: Scattering of two (pseudo)scalars (|J,M> = |0,0>)
  df['J'] = [(0,0)]*len(df.index) 
  df['M'] = [(0,0)]*len(df.index)

#  print df[((df['p'] == tuple([(0,-1,0),(0,1,1)])) | (df['p'] == tuple([(0,1,1),(0,-1,0)])) | (df['p'] == tuple([(0,1,0),(0,-1,1)])) | (df['p'] == tuple([(0,-1,1),(0,1,0)])) | (df['p'] == tuple([(-1,0,0),(1,0,1)])) | (df['p'] == tuple([(1,0,1),(-1,0,0)])) | (df['p'] == tuple([(1,0,0),(-1,0,1)])) | (df['p'] == tuple([(-1,0,1),(1,0,0)]))) & df['cg-coefficient'] != 0]
#  print df['Irrep'].unique()
#  print df[(df['Irrep'] == 'Ep1g')]

  # we want all possible irreps for pipi, only the combinations possible for
  # C2 for the rho
  #df = select_irrep(df, irrep)

  return df

def read_sc_2(p_cm_vecs, path, verbose=True, j=1):
  """
  Read subduction coefficients from SO(3) to irreducible representations of 
  appropriate little group of rotational symmetry for lattice in a reference 
  frame moving with momentum p_cm \in list_p_cm

  Parameters
  ----------
    p_cm_vecs : list
        Center of mass momentum of the lattice. Used to specify the appropriate
        little group of rotational symmetry. Contains integer 3-vectors
    path : string
        Path to files with subduction coefficients

  Returns
  -------
    df : pd.DataFrame
        Contains subduction coefficients for going from continuum to discete
        space. 
        Has columns Irrep, mult, J, M, coefficient, p, \mu and unnamed 
        indices

  Note
  ----
    Filename of subduction coefficients hardcoded. Expected to be 
    "J%d-P%1i%1i%1i-operators.txt"
    # "lattice-basis_J%d_P%1i%1i%1i_Msum.dataframe"
  """

  subduction_coefficients = DataFrame()

  for p_cm_vec in p_cm_vecs:
    name = path +'/' + 'J{0}-P{1}-operators.txt'.format(\
           j, "".join([str(p) for p in eval(p_cm_vec)]))

    if not os.path.exists(name):
      print 'Warning: Could not find {}'.format(name)
      continue

    df = pd.read_csv(name, sep="\t", dtype=str)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    df['p_{cm}'] = [p_cm_vec] * len(df)
    df.rename(columns={'alpha' : '\mu'}, inplace=True)
    del df['beta']
    df['mult'] = 1
    df = df.set_index(['p_{cm}', 'Irrep', '\mu', 'mult'])

    df['coefficient'] = df['coefficient'].apply(aeval)
    df['p^1'] = df['p^1'].apply(literal_eval).apply(str)
    df['p^2'] = df['p^2'].apply(literal_eval).apply(str)
    df['q'] = df['q'].apply(literal_eval).apply(str)
    df.rename(columns={'p^1' : 'p^{0}', 'p^2' : 'p^{1}'}, inplace=True)
    df = df[(df['p^{0}'] != str((0,0,0))) | (df['p^{1}'] != str((0,0,0)))]
    del df['abs(p1)']
    del df['abs(p2)']

    df['J^{0}'] = 0
    df['J^{1}'] = 0
    df['M^{0}'] = 0
    df['M^{1}'] = 0

    if verbose:
      print 'subduction_coefficients for {}'.format(p_cm_vec)
      print df, '\n'

    subduction_coefficients = pd.concat([subduction_coefficients, df])

  return subduction_coefficients.sort_index()


# TODO: Write a function to calculate cross product if basis is not 
#       complete and orthonormalize basis states
# To extend to 2 particle operators this must support use of 2 momenta as well.
# We get an additional column \vec{q} that works similar to the row of the 
# irrep. Either the maple script has to find unique operators, or an additional
# unique function must be used here. I prefer the former.
def read_sc(p_cm_vecs, path, verbose=True, j=1):
  """
  Read subduction coefficients from SO(3) to irreducible representations of 
  appropriate little group of rotational symmetry for lattice in a reference 
  frame moving with momentum p_cm \in list_p_cm

  Parameters
  ----------
    p_cm_vecs : list
        Center of mass momentum of the lattice. Used to specify the appropriate
        little group of rotational symmetry. Contains integer 3-vectors
    path : string
        Path to files with subduction coefficients

  Returns
  -------
    df : pd.DataFrame
        Contains subduction coefficients for going from continuum to discete
        space. 
        Has columns Irrep, mult, J, M, coefficient, p, \mu and unnamed 
        indices

  Note
  ----
    Filename of subduction coefficients hardcoded. Expected to be 
    "J%d-P%1i%1i%1i-operators.txt"
    # "lattice-basis_J%d_P%1i%1i%1i_Msum.dataframe"
  """

  subduction_coefficients = DataFrame()

  for p_cm_vec in p_cm_vecs:

#    name = path +'/' + 'lattice-basis_J{0}_P{1}_Msum.dataframe'.format(\
#           j, "".join([str(p) for p in eval(p_cm_vec)]))
    name = path +'/' + 'J{0}-P{1}-operators.txt'.format(\
           j, "".join([str(p) for p in eval(p_cm_vec)]))

    if not os.path.exists(name):
      print 'Warning: Could not find {}'.format(name)
      continue
    df = pd.read_csv(name, delim_whitespace=True, dtype=str) 

    df = pd.merge(df.ix[:,2:].stack().reset_index(level=1), df.ix[:,:2], 
                  left_index=True, 
                  right_index=True)

    # Munging of column names
    df.columns = ['M^{0}', 'coefficient', 'Irrep', '\mu']
    df['mult'] = 1
    df['p_{cm}'] = [p_cm_vec] * len(df)
    df['coefficient'] = df['coefficient'].apply(aeval)
    df['M^{0}'] = df['M^{0}'].apply(int)
    df = df.set_index(['p_{cm}', 'Irrep', '\mu', 'mult'])
    df['p^{0}'] = [p_cm_vec] * len(df)
    df['J^{0}'] = j
    df = df[['p^{0}','J^{0}','M^{0}','coefficient']]

    if verbose:
      print 'subduction_coefficients for {}'.format(p_cm_vec)
      print df, '\n'

    subduction_coefficients = pd.concat([subduction_coefficients, df])

  return subduction_coefficients.sort_index()


