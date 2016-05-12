#!/usr/bin/python

import os, errno, math, random, struct
import numpy as np

import IOcontraction
import boot

bootstrapsize = 500
T_test = 48
nb_cnfg_test = 246

def read_and_boot(infile, outfile):
  pi, T, nb_cnfg = IOcontraction.extract_corr_liuming(infile)
  C4 = boot.sym_and_boot(pi, T, nb_cnfg, bootstrapsize, path=outfile)
  if (T is not T_test) or (nb_cnfg is not nb_cnfg_test):
    print "Problem, data length is corrupted in ", infile

################################################################################
# reading two-pint correlation function and bootstrapping it
read_and_boot('../data/pi_corr_p0.dat', 'bootdata/C2_pi_p0')
read_and_boot('../data/pi_corr_p1.dat', 'bootdata/C2_pi_p1')
read_and_boot('../data/pi_corr_p2.dat', 'bootdata/C2_pi_p2')
read_and_boot('../data/pi_corr_p3.dat', 'bootdata/C2_pi_p3')



################################################################################
# CM
read_and_boot('../data/rho_corr_TP0_00.dat', 'bootdata/rho_corr_TP0_00')
read_and_boot('../data/rho_corr_TP0_01.dat', 'bootdata/rho_corr_TP0_01')
read_and_boot('../data/rho_corr_TP0_11.dat', 'bootdata/rho_corr_TP0_11')

################################################################################
# MV1
read_and_boot('../data/rho_corr_TP1_00.dat', 'bootdata/rho_corr_TP1_00')
read_and_boot('../data/rho_corr_TP1_01.dat', 'bootdata/rho_corr_TP1_01')
read_and_boot('../data/rho_corr_TP1_11.dat', 'bootdata/rho_corr_TP1_11')

################################################################################
# MV2
read_and_boot('../data/rho_corr_TP2_00.dat', 'bootdata/rho_corr_TP2_00')
read_and_boot('../data/rho_corr_TP2_01.dat', 'bootdata/rho_corr_TP2_01')
read_and_boot('../data/rho_corr_TP2_11.dat', 'bootdata/rho_corr_TP2_11')

################################################################################
# MV3
read_and_boot('../data/rho_corr_TP3_00.dat', 'bootdata/rho_corr_TP3_00')
read_and_boot('../data/rho_corr_TP3_01.dat', 'bootdata/rho_corr_TP3_01')
read_and_boot('../data/rho_corr_TP3_11.dat', 'bootdata/rho_corr_TP3_11')

################################################################################
# MV4
read_and_boot('../data/rho_corr_TP4_00.dat', 'bootdata/rho_corr_TP4_00')
read_and_boot('../data/rho_corr_TP4_01.dat', 'bootdata/rho_corr_TP4_01')
read_and_boot('../data/rho_corr_TP4_11.dat', 'bootdata/rho_corr_TP4_11')



