#!/usr/bin/python

import numpy as np

import utils
# add all relevant diagrams with according Clebsch-Gordans

p = 0

diagrams = ['C4+B', 'C4+D']


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
#    # if source momenta equal
#    if (qn_box[0] == qn_dia[0] and qn_box[9] == qn_dia[6]):
#      # if sink momenta equal
#      if (qn_box[3] == qn_dia[3] and qn_box[6] == qn_dia[9]):
#        # add diagrams
#        wickd.append((-2)*data[0][i]+1*data[1][j])
#        qn_wickd.append([qn_box[0], qn_box[1], qn_box[2], \
#                        qn_box[9], qn_box[10], qn_box[11], \
#                        qn_box[3], qn_box[4], qn_box[5], \
#                        qn_box[6], qn_box[7], qn_box[8], \
#               'C4_uuuu__p%1i%1i%1i.d000.g5' % (qn_box[0][0], qn_box[0][1], qn_box[0][2]) + \
#               '_p%1i%1i%1i.d000.g5' % (qn_box[9][0], qn_box[9][1], qn_box[9][2]) + \
#               '__p%1i%1i%1i.d000.g5' % (qn_box[3][0], qn_box[3][1], qn_box[3][2]) + \
#               '_p%1i%1i%1i.d000.g5' % (qn_box[6][0], qn_box[6][1], qn_box[6][2]) + '.dat'])

    # if source momenta equal
    if (qn_box[0] == qn_dia[0] and qn_box[3] == qn_dia[3]):
      # if sink momenta equal
      if (qn_box[6] == qn_dia[6] and qn_box[6] == qn_dia[6]):
        # add diagrams
        wickd.append((-2)*data[0][i]+1*data[1][j])
        qn_wickd.append([qn_box[0], qn_box[1], qn_box[2], \
                         qn_box[3], qn_box[4], qn_box[5], \
                         qn_box[6], qn_box[7], qn_box[8], \
                         qn_box[9], qn_box[10], qn_box[11]])

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
#for i in range(0, qn_wickd.shape[0]):
#  path = './readdata/p%1i/single/%s/%s' % \
#          (p, 'C4', qn_wickd[i][-1])
#  np.save(path, wickd[i])

# write all operators
path = './readdata/p%1i/%s_p%1i' % (p, 'C4', p)
np.save(path, wickd)

# write all quantum numbers
path = './readdata/p%1i/%s_p%1i_quantum_numbers' % (p, 'C4', p)
np.save(path, qn_wickd)

print '\tfinished writing\n'
 
