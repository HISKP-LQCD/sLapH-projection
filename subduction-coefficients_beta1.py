import itertools as it
import numpy as np

from src.projection_interface_maple import *

list_of_pcm = list(it.product((0,1,-1), (0,1,-1), (0,1,-1))) + [(0,0,2), (0,2,0), (2,0,0), (0,0,-2), (0,-2,0), (-2,0,0)]

df = read_sc([str(p) for p in list_of_pcm], './one-meson-operators')

df.reset_index(inplace=True)
df.rename(columns={'Irrep' : 'Gamma', 'mult' : 'n', 'p_{cm}' : 'P', '\mu' : 'alpha', r'\beta' : 'beta', 'M^{0}' : 'm', 'coefficient' : 'value', 'J^{0}' : 'l'}, inplace=True)

df = df[df['beta'] == '1']
del df['p^{0}']
del df['beta']

df['Re'] = df['value'].apply(np.real)
df['Im'] = df['value'].apply(np.imag)

del df['value']

df.to_csv('subduction-coefficients_beta1.tsv', sep='\t')
