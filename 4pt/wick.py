#!/usr/bin/python

import numpy as np
import itertools as it

import operator
import collections

import utils
# add all relevant diagrams with according Clebsch-Gordans

p = 0
p_max = 4

diagrams = ['C4+B', 'C4+D']

#def scalar_mul(x, y):
#  return sum(it.imap(operator.mul, x, y))
#
#def abs2(x):
#  if isinstance(x, collections.Iterator):
#    x = list(x)
#  return scalar_mul(x, x)
#
#
#def unique_everseen(iterable, key=None):
#  "List unique elements, preserving order. Remember all elements ever seen."
#  # unique_everseen('AAAABBBCCDAABBB') --> A B C D
#  # unique_everseen('ABBCcAD', str.lower) --> A B C D
#  seen = set()
#  seen_add = seen.add
#  if key is None:
#    for element in it.ifilterfalse(seen.__contains__, iterable):
#      seen_add(element)
#      yield element
#  else:
#    for element in iterable:
#      k = key(element)
#      if k not in seen:
#        seen_add(k)
#        yield element
#  
#lookup_p3 = it.ifilter(lambda x: abs2(x) <= p_max, \
#                       it.product(range(-p_max, p_max+1), repeat=3))
#a = it.ifilter(lambda (x,y): abs2(it.imap(operator.add, x, y)) == p \
#               # for zero center of mass momentum omit the case were both 
#               # particles are at rest (s-wave)
#               and not (p == 0 and tuple(x) == tuple(y) ) \
#    ,it.product(lookup_p3, repeat=2))
#
#b = unique_everseen(it.starmap(lambda x, y: (abs2(x), abs2(y)), a))
#
##TODO: for p != 0 there are equivalent combinations with p1 <-> p2
##TODO: criitical for moving frames only group by summed 3-momentum at source and sink
#for c in b:
#  print c

wickd = []
qn_wickd = []

data = []
qn_data = []
for diagram in diagrams:
  path = './readdata/p%1i/%s_p%1i.npy' % (p, diagram, p)
  data.append(np.load(path))
  path = './readdata/p%1i/%s_p%1i_quantum_numbers.npy' % (p, diagram, p)
  qn_data.append(np.load(path))
  if ( (qn_data[-1].shape[0] != data[-1].shape[0])):
    print '\tBootstrapped operators do not aggree with expected operators'
    exit(0)

for i, qn_box in enumerate(qn_data[0]):
  for j, qn_dia in enumerate(qn_data[1]):
    # if source momenta equal
    if (qn_box[0] == qn_dia[0] and qn_box[9] == qn_dia[6]):
      # if sink momenta equal
      if (qn_box[3] == qn_dia[3] and qn_box[6] == qn_dia[9]):
        # add diagrams
        wickd.append(2*data[0][i]-2*data[1][j])
        qn_wickd.append([qn_box[0], qn_box[1], qn_box[2], \
                        qn_box[9], qn_box[10], qn_box[11], \
                        qn_box[3], qn_box[4], qn_box[5], \
                        qn_box[6], qn_box[7], qn_box[8], \
               'C4_uuuu__p%1i%1i%1i.d000.g5' % (qn_box[0][0], qn_box[0][1], qn_box[0][2]) + \
               '_p%1i%1i%1i.d000.g5' % (qn_box[9][0], qn_box[9][1], qn_box[9][2]) + \
               '__p%1i%1i%1i.d000.g5' % (qn_box[3][0], qn_box[3][1], qn_box[3][2]) + \
               '_p%1i%1i%1i.d000.g5' % (qn_box[6][0], qn_box[6][1], qn_box[6][2]) + '.dat'])

wickd = np.asarray(wickd)
print wickd.shape
qn_wickd = np.asarray(qn_wickd)
print qn_wickd.shape

################################################################################
# write data to disc

utils.ensure_dir('./readdata')
utils.ensure_dir('./readdata/p%1i' % p)
utils.ensure_dir('./readdata/p%1i/single' % p)
utils.ensure_dir('./readdata/p%1i/single/%s' % (p, 'C4'))
# write every operator seperately
for i in range(0, qn_wickd.shape[0]):
  path = './readdata/p%1i/single/%s/%s' % \
          (p, 'C4', qn_wickd[i][-1])
  np.save(path, wickd[i])

# write all operators
path = './readdata/p%1i/%s_p%1i' % (p, 'C4', p)
np.save(path, wickd)

# write all quantum numbers
path = './readdata/p%1i/%s_p%1i_quantum_numbers' % (p, 'C4', p)
np.save(path, qn_wickd)

print '\tfinished writing\n'
 
