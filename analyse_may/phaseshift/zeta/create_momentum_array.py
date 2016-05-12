import math
import numpy as np
import os

# checks if the directory where the file will be written does exist
################################################################################
def ensure_dir(f):
  d = os.path.dirname(f)
  if not os.path.exists(d):
    os.makedirs(d)

# Gives an array of 3d vectors for summation
################################################################################
def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

# creates the momentum array used for the sums
################################################################################
def create_momentum_array(p):
  # these momentum suqares do not exist
  exclude = [7, 15, 23, 28, 31, 39, 47, 55, 60, 63, 71, 79, 87, 92, 95, 103, \
             111, 112, 119, 124, 127, 135, 143, 151, 156, 159, 167, 175, 183,\
             188, 191, 199, 207, 215, 220, 223, 231, 239, 240, 247, 252, 255,\
             263, 271, 279, 284, 287, 295]
  if p in exclude:
    return [], p
  i = int(math.sqrt(p)+1)
  n = [j for j in range(-i,i+1)]
  r = cartesian((n, n, n))
  out = []
  for rr in r:
    if (np.dot(rr, rr) == p):
      out.append(np.ndarray.tolist(rr))
  out = np.asarray(out, dtype=float)
  if p > 302:
    print('cannot converge, see zeta.py - create_momentum_array')
    exit(0)
  return out, p


# creating the momentum arrays and writing them to disk
################################################################################
def main():
  r = create_momentum_array(0)
  for i in range(1, 302):
    r = np.vstack((r, create_momentum_array(i)))
  ensure_dir("./momenta")
  np.save("./momenta", r)

# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
  main()
