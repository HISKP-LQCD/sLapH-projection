#!/usr/bin/python

import IOcontraction

def read_cnfg_id(path):
  f = open(path, 'r')
  cnfg_ids = []
  for line in f:
    cnfg_ids.append(int(line))
  f.close()
  return cnfg_ids

def write_data_to_file(filename, corr_new, T, L):
  first_line = '%i %i 1 %i 1\n' % (len(corr_new)/T, T, L)
  IOcontraction.ensure_dir(filename)
  f = open(filename, 'w')
  f.write(first_line)
  for corr, t in zip(corr_new, range(0,len(corr_new))):
    f.write('%i %.8f %.8f\n' % (t%T, corr.real, corr.imag))
  f.close()

def eliminate_configs_MF(cnfg_ids_all, cnfg_ids_TP, TP, path, L):
  filename_00 = 'rho_corr_TP%i_00.dat' % TP
  filename_01 = 'rho_corr_TP%i_01.dat' % TP
  filename_11 = 'rho_corr_TP%i_11.dat' % TP
  corr_00, T_00, nb_cnfg_00 = IOcontraction.extract_corr_liuming2(path + filename_00)
  corr_01, T_01, nb_cnfg_01 = IOcontraction.extract_corr_liuming2(path + filename_01)
  corr_11, T_11, nb_cnfg_11 = IOcontraction.extract_corr_liuming2(path + filename_11)
  if (T_00 != T_01) or (T_00 != T_11):
    print 'temporal extend is not the same'
  if (nb_cnfg_00 != nb_cnfg_01) or (nb_cnfg_00 != nb_cnfg_11):
    print 'number of configurations is not the same'
  corr_new_00, corr_new_01, corr_new_11 = [], [], []
  index = 0
  for ids in cnfg_ids_TP:
    if ids in cnfg_ids_all:
      for t in range(0, T_00):
        corr_new_00.append(corr_00[index*T_00 + t])
        corr_new_01.append(corr_01[index*T_00 + t])
        corr_new_11.append(corr_11[index*T_00 + t])
    index += 1
  write_data_to_file('../data/' + filename_00, corr_new_00, T_00, L)
  write_data_to_file('../data/' + filename_01, corr_new_01, T_00, L)
  write_data_to_file('../data/' + filename_11, corr_new_11, T_00, L)

################################################################################
################################################################################
################################################################################

L = 24
cnfg_min = 600
cnfg_max = 3104
cnfg_del = 8

cnfg_ids_TP0 = read_cnfg_id('/hiskp2/werner/output/config_number/config_number_A60.24_TP0.dat')
cnfg_ids_TP1 = read_cnfg_id('/hiskp2/werner/output/config_number/config_number_A60.24_TP1.dat')
cnfg_ids_TP2 = read_cnfg_id('/hiskp2/werner/output/config_number/config_number_A60.24_TP2.dat')
cnfg_ids_TP3 = read_cnfg_id('/hiskp2/werner/output/config_number/config_number_A60.24_TP3.dat')
cnfg_ids_TP4 = read_cnfg_id('/hiskp2/werner/output/config_number/config_number_A60.24_TP4.dat')

cnfg_ids = []
for ids in cnfg_ids_TP0:
  if (ids in cnfg_ids_TP1) and (ids in cnfg_ids_TP2) and (ids in cnfg_ids_TP3) and (ids in cnfg_ids_TP4):
    cnfg_ids.append(int(ids))

print len(cnfg_ids), ' configurations'

path = '/hiskp2/correlators/A60.24_L24_T48_beta190_mul0060_musig150_mudel190_kappa1632650/ev120/I1/' 
# moving frames ################################################################
eliminate_configs_MF(cnfg_ids, cnfg_ids_TP0, 0, path, L)
eliminate_configs_MF(cnfg_ids, cnfg_ids_TP1, 1, path, L)
eliminate_configs_MF(cnfg_ids, cnfg_ids_TP2, 2, path, L)
eliminate_configs_MF(cnfg_ids, cnfg_ids_TP3, 3, path, L)
eliminate_configs_MF(cnfg_ids, cnfg_ids_TP4, 4, path, L)


# pion 2pt function ############################################################
path = '/hiskp2/correlators/A60.24_L24_T48_beta190_mul0060_musig150_mudel190_kappa1632650/ev120/liuming/'
corr_0, T_00, nb_cnfg_00 = IOcontraction.extract_corr_liuming3(path + 'pi_corr_p0.dat')
corr_1, T_01, nb_cnfg_01 = IOcontraction.extract_corr_liuming3(path + 'pi_corr_p1.dat')
corr_2, T_11, nb_cnfg_11 = IOcontraction.extract_corr_liuming3(path + 'pi_corr_p2.dat')
corr_3, T_11, nb_cnfg_11 = IOcontraction.extract_corr_liuming3(path + 'pi_corr_p3.dat')

cnfg_ids_pi = range(cnfg_min, cnfg_max+1, cnfg_del)
#TODO: eliminate missing configs on A40.20 and A40.24
corr_new_0, corr_new_1, corr_new_2, corr_new_3 = [], [], [], [] 
index = 0
for ids in cnfg_ids_pi:
  if ids in cnfg_ids:
    for t in range(0, T_00):
      corr_new_0.append(corr_0[index*T_00 + t])
      corr_new_1.append(corr_1[index*T_00 + t])
      corr_new_2.append(corr_2[index*T_00 + t])
      corr_new_3.append(corr_3[index*T_00 + t])
  index += 1
write_data_to_file('../data/' + 'pi_corr_p0.dat', corr_new_0, T_00, L)
write_data_to_file('../data/' + 'pi_corr_p1.dat', corr_new_1, T_00, L)
write_data_to_file('../data/' + 'pi_corr_p2.dat', corr_new_2, T_00, L)
write_data_to_file('../data/' + 'pi_corr_p3.dat', corr_new_3, T_00, L)












