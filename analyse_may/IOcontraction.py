#!/usr/bin/python

import os, errno, struct, tarfile

################################################################################
#extract correlation function from tar file
def extract_corr_tar(name_tar, name_file, cnfg_min, cnfg_max, cnfg_del, T, \
                     verbose = 0): 
  re = []
  im = []
  nb_cnfg = 0 
  for i in range(cnfg_min, cnfg_max+1, 4):
    pidata = tarfile.open(name_tar)
    filename = name_file + '%04d.dat' % i
    if verbose:
      print 'reading data from:', filename
    try:
      data = pidata.extractfile(filename)
      nb_cnfg += 1
    except KeyError:
      print "There is no configuration:", i
    j = 0
    for line in data:
      if j is not 0:
        segments = line.split()
        re.append(float(segments[1]))
        im.append(float(segments[2]))
      j += 1 # skip first line 
  print 'total number of files read in:', nb_cnfg
  corr = [complex(0.0, 0.0)]*nb_cnfg*T
  t = 0
  for x, y in zip(re, im):
    corr[(t%T)*nb_cnfg + t/T] = complex(x, y)
    t += 1
  return corr, nb_cnfg

################################################################################
# extract a correlation function from Liumings format
def extract_corr_liuming(name=''):
  f = open(name, "rb") # Open a file
  first_line = f.readline()
  lattice_data = first_line.split()
  nb_cnfg = int(lattice_data[0])
  T = int(lattice_data[1])
  re = []
  for n in range(0, nb_cnfg):
    for t in range(0, T):
      re.append(float(f.readline().split()[1]))
  corr = [float(0.0)]*nb_cnfg*T
  t = 0
  for x in re:
    corr[(t%T)*nb_cnfg + t/T] = x
    t += 1
  f.close()
  return corr, T, nb_cnfg

################################################################################
# extract a correlation function from Liumings format and do not process it
def extract_corr_liuming2(name=''):
  f = open(name, "rb") # Open a file
  first_line = f.readline()
  lattice_data = first_line.split()
  nb_cnfg = int(lattice_data[0])
  T = int(lattice_data[1])
  corr = []
  for n in range(0, nb_cnfg):
    for t in range(0, T):
      line_split = f.readline().split()
      corr.append(complex(float(line_split[1]), float(line_split[2])))
  f.close()
  return corr, T, nb_cnfg
################################################################################
# extract a correlation function from Liumings format and do not process it
def extract_corr_liuming3(name=''):
  f = open(name, "rb") # Open a file
  first_line = f.readline()
  lattice_data = first_line.split()
  nb_cnfg = int(lattice_data[0])
  T = int(lattice_data[1])
  corr = []
  for n in range(0, nb_cnfg):
    for t in range(0, T):
      line_split = f.readline().split()
      corr.append(complex(float(line_split[1]), float(0.0)))
  f.close()
  return corr, T, nb_cnfg

################################################################################
# extract a correlation function and resorts it
def extract_corr_fct(name='', cnfg_min=0, cnfg_del=0, nb_cfg=0, T=0, gamma=5,\
                     verbose = 0): 
  re = []
  im = []
  for x in range(cnfg_min, cnfg_min+cnfg_del*nb_cfg, cnfg_del):
    filename = name + "%04d" % x + '.dat'
    try:
      f = open(filename, "rb") # Open a file
    except IOError:
      continue
    if verbose:
      print "reading from file: ", f.name
    f.seek(2*8*T*gamma)
    for t in range(0, T):
      re.insert(t, struct.unpack('d', f.read(8))) # returns a tuple -> convers.
      im.insert(t, struct.unpack('d', f.read(8))) # returns a tuple -> convers.
    f.close(); # close the file  
  # conversion of the tuple to list and reorganise
  corr = [complex(0.0, 0.0)]*nb_cfg*T
  t = 0
  for x, y in zip(re, im):
    corr[(t%T)*nb_cfg + t/T] = complex(x[0], y[0])
    t += 1
  return corr

################################################################################
# checks if the directory where the file will be written does exist
def ensure_dir(f):
  d = os.path.dirname(f)
  if not os.path.exists(d):
    os.makedirs(d)

################################################################################
# writing the correlation function in new order
def write_corr_fct(name, re_im, corr, verbose = 1):
  ensure_dir(name)
  f = open(name, "wb") # Open a file
  if verbose:
    print "writing to file: ", f.name
  if re_im == 'real':
    for x in corr:
      f.write(struct.pack('d', x.real))
  elif re_im == 'imag':
    for x in corr:
      f.write(struct.pack('d', x.imag))
  else:
    print 'wring re_im -> must be real or imag'
  f.close()

