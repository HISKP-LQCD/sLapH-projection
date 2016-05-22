#!/usr/bin/python

import numpy as np

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.axes as ax

import itertools as it

import utils
##import irreps as representation

################################################################################
T = 48
p = 2

verbose = 0
  
negative = [2,6]

# gamma structure wich shall be averaged. Last entry of gamma must contain all
# names in LaTeX compatible notation for plot labels
gamma_i =   [1, 2, 3, \
             ['\gamma_1', '\gamma_2', '\gamma_3', '\gamma_i']]
gamma_0i =  [10, 11, 12, \
             ['\gamma_0\gamma_1', '\gamma_0\gamma_2', '\gamma_0\gamma_3', \
              '\gamma_0\gamma_i']]
gamma_50i = [15, 14, 13, \
             ['\gamma_5\gamma_0\gamma_1', '\gamma_5\gamma_0\gamma_2', \
              '\gamma_5\gamma_0\gamma_3', '\gamma_5\gamma_0\gamma_i']]

gamma = [gamma_i, gamma_0i, gamma_50i]

latex = {'gi' : '\gamma_i', \
         'g0gi' : '\gamma_0\gamma_i', \
         'g5g0gi' : '\gamma_5\gamma_0\gamma_i', \
         'g5' : '\gamma_5'}



# list of all filled symbols in matplotlib except 'o'
symbol = ['v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', '8']

diagram = 'C4'

################################################################################
# Functions ####################################################################

################################################################################
# computes the mean and the error, and writes both out
def mean_error_print(boot, write = 0):
  mean = np.mean(boot, axis=-1)
  err  = np.std(boot, axis=-1)
  if write:
    for t, m, e in zip(range(0, len(mean)), mean, err):
      print t, m, e
  return mean, err

# computes the mean and the error, and writes both out
def mean_error_print_foreach_row(boot, write = 0):
  mean = np.zeros_like(boot)
  err  = np.zeros_like(boot)
  for i,irrep in enumerate(boot):
    for k1, gevp_row in enumerate(irrep):
      for k2, gevp_col in enumerate(gevp_row):
        for r,row in enumerate(gevp_col):
          mean[i,k1,k2,r] = np.mean(row, axis=-1)
          err[i,k1,k2,r]  = np.std(row, axis=-1)
  if write:
    for t, m, e in zip(range(0, len(mean)), mean, err):
      print t, m, e
  return mean, err

################################################################################
# mass computation
def compute_mass(boot):
  # creating mass array from boot array
  mass = np.empty(boot[...,0,0].shape + (boot.shape[-2]-1, boot.shape[-1]), dtype=float)
  # computing the mass via formula
  for b in range(0, mass.shape[-1]):
    row = boot[...,b]
    for t in range(1, row.shape[-1]-1):
      mass[...,t-1,b] = (row[...,t-1] + row[...,t+1])/(2.0*row[...,t])
  mass = np.arccosh(mass)
  mean, err = mean_error_print(mass)
  return mass, mass[...,0], err

################################################################################
def symmetrize(data, sinh):
  T = data.shape[1]
  if len(sinh) != data.shape[0]:
    print 'in symmetrize: sinh must contain the correct sign for every ' \
          'correlator'
    exit(0)
  if data.ndim != 3:
    print 'in symmetrize: data must have 3 dimensions'
    exit(0)
  sym = np.zeros((data.shape[0], T/2, data.shape[2]))
  # loop over gevp matrix elements
  for mat_el in range(sym.shape[0]):
    for t in range(1,T/2):
      sym[mat_el,t] = data[mat_el,t] + sinh[mat_el] * data[mat_el,T-t]
    sym[mat_el,0] = data[mat_el, 0]

  return sym


################################################################################
# plotting functions ###########################################################
################################################################################

################################################################################
# plots all correlators into  one plot for each operator. Real and imaginary 
# part into same plot with their own axes
def plot_single(mean_real, err_real, mean_imag, err_imag, qn, pdfplot):
  for op in range(0, qn.shape[0]):
    if verbose:
      print 'plot_single op %i' % op
    plt.title(r'$p_{so} = (%i,%i,%i) \ d_{so} = (%i,%i,%i) \ g_{so} = %2i ' \
               '\quad p_{si} = (%i,%i,%i) \ d_{si} = (%i,%i,%i) \ ' \
               'g_{si} = %2i$' % \
               (qn[op][0][0], qn[op][0][1], qn[op][0][2], qn[op][1][0], \
                qn[op][1][1], qn[op][1][2], qn[op][2], qn[op][3][0], \
                qn[op][3][1], qn[op][3][2], qn[op][4][0], qn[op][4][1], \
                                                     qn[op][4][2], qn[op][5]), \
              fontsize=12)
  
    plt.axis('off')
    plt.subplots_adjust(right=0.75)
   
    host = host_subplot(111, axes_class=AA.Axes)
    
    # create parasite plots with their own axes for real and imaginary part
    par1 = host.twinx()
    par2 = host.twinx()
    offset = 60
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right", axes=par2, \
                                                             offset=(offset, 0))
    par2.axis["right"].toggle(all=True)
    host.set_xlim(-1, mean_real.shape[1]+1)
    host.set_yticklabels([])
    par1.axis["right"].label.set_color(cmap_real[op])
    par2.axis["right"].label.set_color(cmap_imag[op])
  
    # set labels
    host.set_xlabel(r'$t/a$', fontsize=12)
    host.set_ylabel(r'$C_2^0(t/a)$', fontsize=12)
    par1.set_ylabel('real', fontsize=12)
    par2.set_ylabel('imag', fontsize=12)

    # plotting
    real = par1.errorbar(range(0, mean_real.shape[1]), mean_real[op], \
                         err_real[op], fmt=symbol[op%len(symbol)], \
                         color=cmap_real[op], label='real', markersize=3, \
                         capsize=3, capthick=0.5, elinewidth=0.5, \
                                 markeredgecolor=cmap_real[op], linewidth='0.0')
    imag = par2.errorbar(range(0, mean_imag.shape[1]), mean_imag[op], \
                         err_imag[op], fmt=symbol[op%len(symbol)], \
                         color=cmap_imag[op], label='imag', markersize=3, \
                         capsize=3, capthick=0.5, elinewidth=0.5, \
                                 markeredgecolor=cmap_imag[op], linewidth='0.0')
  
    pdfplot.savefig()
    plt.clf()

################################################################################
# plot all momenta for a given gamma combination
def plot_grouped_2(mean_real, err_real, mean_imag, err_imag, quantum_numbers, gamma, \
                                                             pdfplot, log=True):

  nb_mom = 12
  shift = 1./3/nb_mom

  # loop over all irreps and momenta to find the subduction coefficient. Plot
  # all correlator subducing into a given irrep and gamma structure
  mean = np.zeros(mean_real[0].shape)
  err = np.zeros(err_real[0].shape)
  cmap_brg = plt.cm.brg( np.asarray( range(0,nb_mom)) * 256/(nb_mom-1) )
  # operator counter counts how many operators for a given gamma structure
  # are plotted already. Input for cmap
  for g1 in gamma:
    g2 = g1
    for so in range(0,3):
      for si in range(0,3):
        pc = 0 
        for op, qn in enumerate(quantum_numbers):
          # if operator from quantum_numbers loop has the correct gamma
          # structure
          if ( (g1[so] == qn[2]) and (g2[si] == qn[5]) ):

            if (g1[so] in gamma_i) != (g2[si] in gamma_i):
              mean = mean_imag[op]
              err =  err_imag[op]
            else:
              mean = mean_real[op]
              err =  err_real[op]

            # set plot title, labels etc.
            plt.title(r'$p = %i \quad %s - %s$ ' % \
                       (p, g1[-1][so], g2[-1][si]), \
                                                            fontsize=12)
            # set labels
            plt.xlabel(r'$t/a$', fontsize=12)
            plt.ylabel(r'$C_2^0(t/a)$', fontsize=12)
  
            label = r'$p_{so} = (%i,%i,%i) \quad $' % \
                     (qn[0][0], qn[0][1], qn[0][2])
            if log:
              plt.yscale('log')
      
#            plt.errorbar(np.asarray(range(0,mean.shape[-1]))+pc*shift, 
            plt.errorbar(np.asarray(range(3,19))+pc*shift, 
                mean[3:19], err[3:19], fmt=symbol[pc%len(symbol)], \
                         color=cmap_brg[pc], label=label, markersize=1,\
                         capsize=1, capthick=0.4, elinewidth=0.4, \
                          markeredgecolor=cmap_brg[pc], linewidth='0.0')
            pc = pc + 1

        plt.legend(numpoints=1, loc=1, fontsize=6) #bbox_to_anchor=(1.05,1.05))
        pdfplot.savefig()
        plt.clf()

  return 

################################################################################
# plot all gamma combinations entering the subduction into one plot

# for momentum p0 subduction goes into T1. The necessary gamma combinations are 
# g1g1, g2g2, g3g3 and terms mixing with other gamma structures
# TODO: plot whole correlator log and make picture in picture for the non-log 
# timeslices 8:15
# TODO: get legend out of picture to the right side
# TODO: make fit interval variable
def plot_grouped(mean_real, err_real, mean_imag, err_imag, quantum_numbers, gamma, \
                                                             pdfplot, log=True):

  if p in [0]:
    irreps = [['T1']]
  elif p in [1, 3, 4]:
    irreps = [['A1', 'E2_1', 'E2_2']]
  elif p in [2]:
    irreps = [['A1', 'B1', 'B2']]

  # get factors for the desired irreps
  for i in irreps[-1]:
    irreps.insert(-1,representation.coefficients(i))

  nb_mom = 0
  for el in irreps[0]:
    if( np.dot(el[0], el[0]) == p):
      nb_mom = nb_mom + 1
  print 'nb_mom = %i' % nb_mom

  shift = 1./3/nb_mom

  # loop over all irreps and momenta to find the subduction coefficient. Plot
  # all correlator subducing into a given irrep and gamma structure
  mean = np.zeros(mean_real[0].shape)
  err = np.zeros(err_real[0].shape)
  for i, irrep in enumerate(irreps[:-1]):
    if verbose:
      print 'plotting %s irrep' % irreps[-1][i]
    cmap_brg = plt.cm.brg( np.asarray( range(0,nb_mom)) * 256/(nb_mom-1) )
    # operator counter counts how many operators for a given gamma structure
    # are plotted already. Input for cmap
    for g1 in gamma:
      for g2 in gamma:
        oc = 0 
        pc = 0
        for el in irrep:
          for op, qn in enumerate(quantum_numbers):
            # if source momentum was found in irreps
            if np.array_equal(el[0], qn[0]):
              # TODO: check for gamma correct g1, g2, qn
              for so in range(0,3):
                for si in range(0,3):
                  # if operator from quantum_numbers loop has the correct gamma
                  # structure
                  if ( (g1[so] == qn[2]) and (g2[si] == qn[5]) ):

                    # Hardcoded 2pt functions here because 
                    #   sink operator = conj(source operator)
                    # and irreps are diagonal. Otherwise el[g_si] must be 
                    # calculated in seperate loop over the sink momentum
                    factor = el[so+1] * np.conj(el[si+1])
                    if (factor == 0):
                      continue
                    # calculating the Correlators yields imaginary parts in 
                    # gi-g0gi, gi-gjgk and cc. Switching real and imaginary
                    # part in the subduction factor accounts for this.
                    if (g1[so] in gamma_i) != (g2[si] in gamma_i):
                      factor = factor.imag + factor.real * 1j
  #                    factor = (1j) * factor
  
                    if factor.real != 0:
                      mean = factor.real*mean_real[op]
                      err =  factor.real*err_real[op]
                    elif factor.imag != 0:
                      mean = factor.imag*mean_imag[op]
                      err =  factor.imag*err_imag[op]
                    else:
                      print 'Something went wrong in the factor calculation ' \
                            'for g_so = %i - g_si = %i in the %s irrep' % \
                            (g1[so], g2[si], irreps[-1,i])
 
                    # TODO: refactor that somewhere else
                    j = 3*so+si
                    if j in [0,2,3,4,5,7,8]:
                      mean = (-2) * mean
                      err  = (-2) * err
                    else:
                      mean = 2 * mean
                      err  = 2 * err
                    # end of TODO

                    # set plot title, labels etc.
                    plt.title(r'$p = %i \quad g_{so} = %s - g_{si} = %s$ ' \
                               'Correlators contributing to $%s$ irrep' % \
                               (p, g1[-1][-1], g2[-1][-1], irreps[-1][i]), \
                                                                    fontsize=12)
                    # set labels
                    plt.xlabel(r'$t/a$', fontsize=12)
                    plt.ylabel(r'$C_2^0(t/a)$', fontsize=12)
  
                    label = r'$p_{so} = (%i,%i,%i) \quad g_{so} = %s - ' \
                             'g_{si} = %s$' % \
                             (qn[0][0], qn[0][1], qn[0][2], g1[-1][so], \
                                                                     g2[-1][si])
                    if log:
                      plt.yscale('log')
            
                    plt.errorbar(np.asarray(range(0,mean.shape[-1]))+oc*shift, 
                                 mean, err, fmt=symbol[oc%len(symbol)], \
                                 color=cmap_brg[pc], label=label, markersize=1,\
                                 capsize=1, capthick=0.4, elinewidth=0.4, \
                                  markeredgecolor=cmap_brg[pc], linewidth='0.0')
                    oc = oc + 1

          if( np.dot(el[0], el[0]) == p):
            pc = pc + 1
   
        plt.legend(numpoints=1, loc=1, fontsize=6) #bbox_to_anchor=(1.05,1.05))
        pdfplot.savefig()
        plt.clf()

  return 

################################################################################
def plot_avg(mean_sin, err_sin, qn_sin, pdfplot):

  cmap_brg = plt.cm.brg(np.asarray(range(0, qn_sin.shape[2])) * \
                                   256/(qn_sin.shape[2]-1))

  for i, irrep in enumerate(qn_sin):
    for r, row in enumerate(irrep):
      print 'plot row %i of irrep %s' % (r, irrep[r,0,-1])

      shift = 1./3/row.shape[0]

      # set plot title, labels etc.
#      plt.title(r'$%s$ - $%s$ Operators subduced into $p = %i$ under $%s$ ' \
#                 'irrep' % \
#                  (irrep_gamma[0,-3], irrep_gamma[0,-2], p, irrep_gamma[0,-1]),\
#                fontsize=12)
#      plt.xlabel(r'$t/a$', fontsize=12)
#      plt.ylabel(r'$C_2^0(t/a)$', fontsize=12)
  
      for so in [1,2]:
        for si in [1,2]:
          for op in range(0, row.shape[0]):
            if np.sum(np.square(row[op][0])) == so and np.sum(np.square(row[op][1])) == so:
              if np.sum(np.square(row[op][2])) == si and np.sum(np.square(row[op][3])) == si:
  
        #        label = r'$p_{so} = (%2i,%2i,%2i)$' % (np.int(irrep_gamma[op][0]), \
        #                         np.int(irrep_gamma[op][1]), np.int(irrep_gamma[op][2]))
        #        label = r'$\mu = %i$' % op
                label = r'$p_{so} = (%2i,%2i,%2i), (%2i,%2i,%2i)  ' \
                          'p_{si} = (%2i,%2i,%2i), (%2i,%2i,%2i)$' % \
                          (row[op][0][0], row[op][0][1], row[op][0][2], \
                           row[op][1][0], row[op][1][1], row[op][1][2], \
                           row[op][2][0], row[op][2][1], row[op][2][2], \
                           row[op][3][0], row[op][3][1], row[op][3][2])
        
                # prepare data for plotting
                # TODO: put that in subduction
                plot = (-1)*mean_sin[i,r,op,:23]
        #        if j == 1 or j == 6:
        #          plot = 2*mean_sin[i,j,op,:23]
               
                plt.yscale('log')
                # plotting single correlators subduced into irrep
        #        plt.errorbar(np.asarray(range(0, mean_sin.shape[-1]))+op*shift, plot, err_sin[i,j,op], \
                plt.errorbar(np.asarray(range(0, 23))+op*shift, plot, err_sin[i,r,op,:23], \
                             fmt=symbol[op%len(symbol)], color=cmap_brg[op], \
                             label=label, markersize=3, capsize=3, capthick=0.5, \
                             elinewidth=0.5, markeredgecolor=cmap_brg[op], \
                                                                        linewidth='0.0')

#      if plot_mean == True:
#        # prepare data for plotting
#        plot = -2*mean_avg[i,j]
#        if j == 1 or j == 6:
#          plot = 2*mean_avg[i,j]
#  
#        plt.yscale('log')
#  
#        # plotting average for irrep
#        plt.errorbar(range(0, mean_avg.shape[-1]), plot, err_avg[i,j], \
#                     fmt='o', color='black', label=r'$avg$', \
#                     markersize=3, capsize=3, capthick=0.75, elinewidth=0.75, \
#                                      markeredgecolor='black', linewidth='0.0')
       
          plt.legend(numpoints=1, loc=1, fontsize=6)
          pdfplot.savefig()
          plt.clf()
  
#  if plot_mean == True:
#    # plot averages from all irreps into one plot
#    cmap_brg = plt.cm.brg(np.asarray(range(0, qn_avg.shape[0]*qn_avg.shape[1])) *\
#                                     224/(qn_avg.shape[0]*qn_avg.shape[1]-1))
#  
#    plt.title(r'Averages of operators subduced into $p = %i$' % p, fontsize=12)
#    plt.xlabel(r'$t/a$', fontsize=12)
#    plt.ylabel(r'$C_2^0(t/a)$', fontsize=12)
#  
#    for i in range(0, qn_avg.shape[0]):
#      for j in range(0, qn_avg.shape[1]):
#        # index for cmap
#        c = i*qn_avg.shape[1]+j
#        label = r'$%s \ %s - %s$' % \
#                                  (qn_avg[i,j,-1], qn_avg[i,j,-3], qn_avg[i,j,-2])
#  
#        # in overview plot only plot diagonal elements
#        if j not in [0,4,8]:
#          continue
#  
#        # prepare data for plotting
#        plot = -2*mean_avg[i,j]
#        if j == 1 or j == 6:
#          plot = 2*mean_avg[i,j]
#  
#        plt.yscale('log')
#        plt.errorbar(range(0, mean_avg.shape[-1]), plot/plot[6], \
#                     err_avg[i,j]/plot[6], fmt='o', color=cmap_brg[c], \
#                     label=label, markersize=3, capsize=3, capthick=0.75, \
#                     elinewidth=0.75, markeredgecolor=cmap_brg[c], \
#                                                                  linewidth='0.0')
#    plt.legend(numpoints=1, loc=5, fontsize=6)
#    pdfplot.savefig()
#    plt.clf()
 
  return



#################################################################################
def plot_avg_and_gevp(mean_avg, err_avg, qn_avg, mean_gevp, err_gevp, qn_gevp, pdfplot):

  n = np.int(np.sqrt(qn_avg.shape[1]))
  diag_indices = [0,4,8]
  if( n != mean_gevp.shape[1] ):
    print 'Subduced operators do not build %ix%i gevp' % (mean_gevp.shape[1], mean_gevp.shape[1])

  cmap_ocean = plt.cm.ocean(np.asarray(range(0, mean_gevp.shape[1])) * 128/(mean_gevp.shape[1]-1))
  cmap_autumn = plt.cm.autumn(np.asarray(range(0, n)) * 128/(n-1))
  for i in range(qn_gevp.shape[0]):
  
    plt.xlabel(r'$t/a$')
    plt.ylabel('Principal Correlator')
    plt.title(r'GEVP $t_0 =$, $p = %d$' % p)

    for j in range(mean_gevp.shape[1]):
      label = '%i. principal correlator' % j
      plt.yscale('log')
  
#      plt.errorbar(range(5, mean_gevp.shape[-1]), mean_gevp[i,j,5:24], err_gevp[i,j,5:24], \
      plt.errorbar(range(5, mean_gevp.shape[-1]), mean_gevp[i,j,5:24]/mean_gevp[i,j,6], err_gevp[i,j,5:24]/mean_gevp[i,j,6], \
                   fmt=symbol[j%len(symbol)], color=cmap_ocean[j], label=label, \
                   markersize=3, capsize=3, capthick=0.5, elinewidth=0.5, \
                                    markeredgecolor=cmap_ocean[j], linewidth='0.0')
  
    for c in range(n):
      # index for cmap
      j = diag_indices[c]
      label = r'$%s %s %s$' % (qn_avg[i,j,-1], qn_avg[i,j,-3], qn_avg[i,j,-2])

      # prepare data for plotting
#      if j%(np.int(np.sqrt(qn_avg.shape[1]))) == 1:
      plot = -2*mean_avg[i,j]
      if j == 1 or j == 3:
        plot = 2*mean_avg[i,j]
#      if j in negative:
#        plot = -mean_avg[i,j]#,5:mean_avg.shape[-1]/2]
#      else:
#        plot = mean_avg[i,j]#,5:mean_avg.shape[-1]/2]
      plt.yscale('log')

#      plt.errorbar(range(5, mean_avg.shape[-1]/2), plot[5:mean_avg.shape[-1]/2], err_avg[i,j,5:mean_avg.shape[-1]/2], \
      plt.errorbar(range(5, mean_avg.shape[-1]/2), plot[5:mean_avg.shape[-1]/2]/plot[6], err_avg[i,j,5:mean_avg.shape[-1]/2]/plot[6], \
                   fmt=symbol[(c+n)%len(symbol)], color=cmap_autumn[c], label=label, \
                   markersize=3, capsize=3, capthick=0.75, elinewidth=0.75, \
                                    markeredgecolor=cmap_autumn[c], linewidth='0.0')
    plt.legend(numpoints=1, loc=3, fontsize=6)
    pdfplot.savefig()
    plt.clf()
 
  return

################################################################################
# plot subduced operators for all momentum directions and mean into one plot
# one plot for each irrep
# TODO: plotting function with and without log
def plot_mass(avg, qn_avg, pdfplot):

  # plot averages from all irreps into one plot
#  cmap_brg = plt.cm.brg(np.asarray(range(0, qn_avg.shape[0]*qn_avg.shape[1])) * \
#                                   224/(qn_avg.shape[0]*qn_avg.shape[1]-1))
  cmap_brg = plt.cm.brg(np.asarray(range(0, 7)) * \
                                   256/(6))

  plt.title(r'Averages of operators subduced into $p = %i$' % p, fontsize=12)
  plt.xlabel(r'$t/a$', fontsize=12)
  plt.ylabel(r'$C_2^0(t/a)$', fontsize=12)

  sinh = [+1, -1, -1, -1, +1, +1, -1, +1, +1]

  for i in range(0, qn_avg.shape[0]):
    for j in range(0, qn_avg.shape[1]):
     # prepare data for plotting
#      if j%(np.int(np.sqrt(qn_avg.shape[1]))) != 0:
#        avg[i,j] = -avg[i,j]
      if j in negative:
        avg[i,j] = -avg[i,j]

#      print 'Correlator'
#      mean_error_print(avg[i,j], True)


    sym = symmetrize(avg[i], sinh)
    print sym.shape

  c = 0
  for i in range(0, qn_avg.shape[0]):
    for j in range(0, qn_avg.shape[1]):
      if j in [3,6,7]:
        continue
      else:
        c = c+1
        # index for cmap

      label = r'$%s \ %s - %s$' % (qn_avg[i,j,-1], qn_avg[i,j,-3], qn_avg[i,j,-2])
#      plt.yscale('log')

      print '%i %i' % (i, j)
      print 'effektive masse'
#      mass_avg, mass_mean_avg, mass_err_avg = compute_mass(sym[j])
      mass_avg, mass_mean_avg, mass_err_avg = compute_mass(sym[j])
      print mass_mean_avg
#      plt.errorbar(range(0, mean_avg.shape[-1]), mean_avg[i]/mean_avg[i,10], err_avg[i], \
#      plt.plot(range(0, 23), mass_mean_avg,  \
#                   'o', color=cmap_brg[c], label=label)

      plt.errorbar(range(0, 23), mass_mean_avg, mass_err_avg, \
                   fmt='o', color=cmap_brg[c], label=label, \
                   markersize=3, capsize=3, capthick=0.75, elinewidth=0.75, \
                                    markeredgecolor=cmap_brg[c], linewidth='0.0')
  plt.legend(numpoints=1, loc=1, fontsize=6)
  pdfplot.savefig()
  plt.clf()
 
  return

################################################################################
# plot all combinations of momenta at source/ sink subducing into the same
# \Lambda, [|\vec{k1}|, |\vec{k2}|], \mu, \vec{P}
def plot_vecks(mean_sin, err_sin, qn_sin, mean_avg, err_avg, pdfplot):

#  ax = plt.subplot(111)

  # TODO: include loop over gamma structure and create additional 
  # array-dimension with just \gamma_5 as entry
  for i, irrep in enumerate(qn_sin):
    for k1, gevp_row in enumerate(irrep):
      for k2, gevp_col in enumerate(gevp_row):
        for r, row in enumerate(gevp_col):
          print 'plot row %i of irrep %s, [%i,%i] -> [%i,%i]' % (r, row[0,-1], \
                 row[0,-5][0], row[0,-5][1], row[0,-4][0], row[0,-4][1])

          cmap_brg = plt.cm.brg(np.asarray(range(0, row.shape[0])) * \
                                           256/(row.shape[0]-1))
          if verbose:
            print row.shape[0] 
          shift = 1./3/row.shape[0]

          # set plot title, labels etc.
          plt.title(r'$%s%s$ - $%s%s$ Operators ' \
                    r'subduced into $p = %i$, $[%i,%i] \ \to \ [%i,%i]$ ' \
                    r'under $\Lambda = %s$ $\mu = %i$' % \
                      (latex[row[0,-3]], latex[row[0,-3]], latex[row[0,-2]], \
                       latex[row[0,-2]], p, row[0][-5][0], row[0][-5][1], \
                       row[0][-4][0], row[0][-4][1], row[0][-1], r+1),\
                    fontsize=12)
          plt.xlabel(r'$t/a$', fontsize=12)
          plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)
 
          for op in range(0, row.shape[0]):

            label = r'$[(%2i,%2i,%2i), (%2i,%2i,%2i)] \ \to \ ' \
                    r'[(%2i,%2i,%2i), (%2i,%2i,%2i)]$' % \
                      (row[op][0][0], row[op][0][1], row[op][0][2], \
                       row[op][1][0], row[op][1][1], row[op][1][2], \
                       row[op][2][0], row[op][2][1], row[op][2][2], \
                       row[op][3][0], row[op][3][1], row[op][3][2])
#            label = '_nolegend_'
            
            # prepare data for plotting
            # TODO: put that in subduction
            mean = mean_sin[i,k1,k2,r][op,:23]
            err = err_sin[i,k1,k2,r][op,:23]
                  
#              # Shrink current axis by 20%
#              box = ax.get_position()
#              ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#              # Put a legend to the right of the current axis
#              ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#            plt.yscale('log')
            plt.errorbar(np.asarray(range(0, 23))+op*shift, mean, err, \
                         fmt=symbol[op%len(symbol)], color=cmap_brg[op], \
                         label=label, markersize=3, capsize=3, capthick=0.5, \
                         elinewidth=0.5, markeredgecolor=cmap_brg[op], \
                                                                    linewidth='0.0')

          mean = mean_avg[i,k1,k2,r,:23]
          err = err_avg[i,k1,k2,r,:23]
          plt.errorbar(np.asarray(range(0, 23))+op*shift, mean, err, \
                       fmt='o', color='black', \
                       label='average', markersize=3, capsize=3, capthick=0.5, \
                       elinewidth=0.5, markeredgecolor='black', \
                                                                  linewidth='0.0')
    
          plt.legend(numpoints=1, loc='best', fontsize=6).get_frame().set_alpha(0.5)
          pdfplot.savefig()
          plt.clf()

  print ' '
  return

################################################################################
# plot all rows for averages over all combinations of momenta at source/ sink 
# subducing into the same \Lambda, [|\vec{k1}|, |\vec{k2}|]
def plot_rows(mean_sin, err_sin, qn_sin, mean_avg, err_avg, pdfplot):

  # TODO: include loop over gamma structure and create additional 
  # array-dimension with just \gamma_5 as entry
  for i, irrep in enumerate(qn_sin):
    for k1, gevp_row in enumerate(irrep):
      for k2, gevp_col in enumerate(gevp_row):
        print 'plot irrep %s, [%i,%i] -> [%i,%i]' % (gevp_col[0,-1], \
               gevp_col[0,-5][0], gevp_col[0,-5][1], gevp_col[0,-4][0], gevp_col[0,-4][1])

        cmap_brg = plt.cm.brg(np.asarray(range(0, gevp_col.shape[0])) * \
                                         256/(gevp_col.shape[0]-1))
        if verbose:
          print gevp_col.shape[0] 
        shift = 1./3/gevp_col.shape[0]
        for op in range(gevp_col.shape[0]):

          #TODO: title
          # set plot title, labels etc.
          plt.title(r'$%s%s$ - $%s%s$ Operators ' \
                    r'subduced into $p = %i$, $[%i,%i] \ \to \ [%i,%i]$ ' \
                    r'under $\Lambda = %s$' % \
                      (latex[row[0,-3]], latex[row[0,-3]], latex[row[0,-2]], \
                       latex[row[0,-2]], p, gevp_col[op][-5][0], \
                       gevp_col[op][-5][1], gevp_col[op][-4][0], \
                       gevp_col[op][-4][1], gevp_col[op][-1]),\
                    fontsize=12)
          plt.xlabel(r'$t/a$', fontsize=12)
          plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)

 
#          if abs(mean_sin[i,k,op,0]) >= abs(err_avg[i,k,0]):
          label = r'$\mu = %i$' % (op+1)
#          else:
#            label = '_nolegend_'
          
          # prepare data for plotting
          # TODO: put that in subduction
          mean = mean_sin[i,k1,k2,op]
          err = err_sin[i,k1,k2,op]
                
          plt.yscale('log')
          plt.errorbar(np.asarray(range(0, mean.shape[-1]))+op*shift, mean, err, \
                       fmt=symbol[op%len(symbol)], color=cmap_brg[op], \
                       label=label, markersize=3, capsize=3, capthick=0.5, \
                       elinewidth=0.5, markeredgecolor=cmap_brg[op], \
                                                                  linewidth='0.0')

        mean = mean_avg[i,k1,k2]
        err = err_avg[i,k1,k2]
        plt.yscale('log')
        plt.errorbar(np.asarray(range(0, mean.shape[-1]))+op*shift, mean, err, \
                     fmt='o', color='black', \
                     label='average', markersize=3, capsize=3, capthick=0.5, \
                     elinewidth=0.5, markeredgecolor='black', \
                                                                linewidth='0.0')
    
        plt.legend(numpoints=1, loc='best', fontsize=6)
        pdfplot.savefig()
        plt.clf()
  
  print ' '
  return

################################################################################
# plot all combinations of momenta at source/ sink subducing into the same
# \Lambda, [|\vec{k1}|, |\vec{k2}|], \mu, \vec{P}
def plot_abs(mean_sin, err_sin, qn_sin, mean_avg, err_avg, pdfplot):

#  ax = plt.subplot(111)

  # TODO: include loop over gamma structure and create additional 
  # array-dimension with just \gamma_5 as entry
  for i, irrep in enumerate(qn_sin):
    for k1, gevp_row in enumerate(irrep):
      for k2, gevp_col in enumerate(gevp_row):
        for r, row in enumerate(gevp_col):
          print 'plot row %i of irrep %s, [%i,%i] -> [%i,%i]' % (r, row[0,-1], \
                 row[0,-5][0], row[0,-5][1], row[0,-4][0], row[0,-4][1])

          cmap_brg = plt.cm.brg(np.asarray(range(0, row.shape[0])) * \
                                           256/(row.shape[0]-1))
          if verbose:
            print row.shape[0] 
          shift = 1./3/row.shape[0]
          for op in range(0, row.shape[0]):

          #TODO: title
            # set plot title, labels etc.
            plt.title(r'$%s%s$ - $%s%s$ Operators ' \
                      r'subduced into $p = %i$, $[%i,%i] \ \to \ [%i,%i]$ ' \
                      r'under $\Lambda = %s$ $\mu = %i$' % \
                        (latex[row[0,-3]], latex[row[0,-3]], latex[row[0,-2]], \
                         latex[row[0,-2]], p, row[op][-5][0], row[op][-5][1], \
                         row[op][-4][0], row[op][-4][1], row[op][-1], r+1),\
                      fontsize=12)
            plt.xlabel(r'$t/a$', fontsize=12)
            plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)
  
#            if abs(mean_sin[i,k,r][op,0]) >= 0.05*abs(np.max(mean_sin[i,k,r][:,0])):
            label = r'$[(%2i,%2i,%2i), (%2i,%2i,%2i)] \ \to \ ' \
                    r'[(%2i,%2i,%2i), (%2i,%2i,%2i)]$' % \
                      (row[op][0][0], row[op][0][1], row[op][0][2], \
                       row[op][1][0], row[op][1][1], row[op][1][2], \
                       row[op][2][0], row[op][2][1], row[op][2][2], \
                       row[op][3][0], row[op][3][1], row[op][3][2])
#            else:
#              label = '_nolegend_'
            
            # prepare data for plotting
            # TODO: takewhile breaks one iteration to early
            mean = it.takewhile(lambda (x,y): x/y > 0, \
                                it.izip(mean_sin[i,k1,k2,r][op,1:22], mean_sin[i,k1,k2,r][op,2:23]))
            mean = np.asarray(list(abs(m[1]) for m in mean))
            mean = np.insert(mean, 0, abs(mean_sin[i,k1,k2,r][op,1]))
            err = err_sin[i,k1,k2,r][op,1:(mean.shape[0]+1)]
                  
#              # Shrink current axis by 20%
#              box = ax.get_position()
#              ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#              # Put a legend to the right of the current axis
#              ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.yscale('log')
            plt.errorbar(np.asarray(range(1, mean.shape[0]+1))+op*shift, mean, err, \
                         fmt=symbol[op%len(symbol)], color=cmap_brg[op], \
                         label=label, markersize=3, capsize=3, capthick=0.5, \
                         elinewidth=0.5, markeredgecolor=cmap_brg[op], \
                                                                    linewidth='0.0')

#          mean = mean_avg[i,k1,k2,r,:23]
#          err = err_avg[i,k1,k2,r,:23]
#          plt.errorbar(np.asarray(range(0, 23))+op*shift, mean, err, \
#                       fmt='o', color='black', \
#                       label='average', markersize=3, capsize=3, capthick=0.5, \
#                       elinewidth=0.5, markeredgecolor='black', \
#                                                                  linewidth='0.0')
      
          plt.legend(numpoints=1, loc='best', fontsize=6).get_frame().set_alpha(0.5)
          pdfplot.savefig()
          plt.clf()

  print ' '
  return

################################################################################
# plot all combinations of momenta at source/ sink subducing into the same
# \Lambda, [|\vec{k1}|, |\vec{k2}|], \mu, \vec{P}
def plot_signal_to_noise(mean_sin, err_sin, qn_sin, mean_avg, err_avg, pdfplot):

#  ax = plt.subplot(111)

  # TODO: include loop over gamma structure and create additional 
  # array-dimension with just \gamma_5 as entry
  for i, irrep in enumerate(qn_sin):
    for k, k1k2 in enumerate(irrep):
      for r, row in enumerate(k1k2):
        print 'plot row %i of irrep %s, [%i,%i] -> [%i,%i]' % (r, row[0,-1], \
               row[0,-5][0], row[0,-5][1], row[0,-4][0], row[0,-4][1])

        cmap_brg = plt.cm.brg(np.asarray(range(0, row.shape[0])) * \
                                         256/(row.shape[0]-1))
        if verbose:
          print row.shape[0] 
        shift = 1./3/row.shape[0]
        for op in range(0, row.shape[0]):

        #TODO: title
          # set plot title, labels etc.
          plt.title(r'$%s%s$ - $%s%s$ Operators ' \
                    r'subduced into $p = %i$, $[%i,%i] \ \to \ [%i,%i]$ ' \
                    r'under $\Lambda = %s$ $\mu = %i$' % \
                      (latex[row[0,-3]], latex[row[0,-3]], latex[row[0,-2]], \
                       latex[row[0,-2]], p, row[op][-5][0], row[op][-5][1], \
                       row[op][-4][0], row[op][-4][1], row[op][-1], r+1),\
                    fontsize=12)
          plt.xlabel(r'$t/a$', fontsize=12)
          plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)
  
#          if abs(mean_sin[i,k,r][op,0]) >= 0.05*abs(np.max(mean_sin[i,k,r][:,0])):
          label = r'$[(%2i,%2i,%2i), (%2i,%2i,%2i)] \ \to \ ' \
                  r'[(%2i,%2i,%2i), (%2i,%2i,%2i)]$' % \
                    (row[op][0][0], row[op][0][1], row[op][0][2], \
                     row[op][1][0], row[op][1][1], row[op][1][2], \
                     row[op][2][0], row[op][2][1], row[op][2][2], \
                     row[op][3][0], row[op][3][1], row[op][3][2])
#          else:
#            label = '_nolegend_'
          
          # prepare data for plotting
          # TODO: put that in subduction
          mean = mean_sin[i,k,r][op,:23]
          err = err_sin[i,k,r][op,:23]
                
#            # Shrink current axis by 20%
#            box = ax.get_position()
#            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#            # Put a legend to the right of the current axis
#            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#          plt.yscale('log')
          plt.errorbar(np.asarray(range(0, 23))+op*shift, mean, err, \
                       fmt=symbol[op%len(symbol)], color=cmap_brg[op], \
                       label=label, markersize=3, capsize=3, capthick=0.5, \
                       elinewidth=0.5, markeredgecolor=cmap_brg[op], \
                                                                  linewidth='0.0')

        mean = mean_avg[i,k,r,:23]
        err = err_avg[i,k,r,:23]
        plt.errorbar(np.asarray(range(0, 23))+op*shift, mean, err, \
                     fmt='o', color='black', \
                     label='average', markersize=3, capsize=3, capthick=0.5, \
                     elinewidth=0.5, markeredgecolor='black', \
                                                                linewidth='0.0')
    
        plt.legend(numpoints=1, loc='best', fontsize=6).get_frame().set_alpha(0.5)
        pdfplot.savefig()
        plt.clf()

  print ' '
  return

################################################################################
# read data

for p in [0]:

  # bootstrapped correlators
  diagram = 'C4'
#  filename = './bootdata/p%1i/%s_p%1i_real.npy' % (p, diagram, p)
#  data = np.load(filename)
#  mean_real, err_real = mean_error_print(data)
#  filename = './bootdata/p%1i/%s_p%1i_imag.npy' % (p, diagram, p)
#  data = np.load(filename)
#  mean_imag, err_imag = mean_error_print(data)
#  
#  print mean_real.shape
#  print mean_imag.shape
#  
#  filename = './bootdata/p%1i/%s_p%1i_quantum_numbers.npy' % (p, diagram,  p)
#  qn = np.load(filename)
#  
#  if ( (qn.shape[0] != mean_real.shape[0]) or \
#       (qn.shape[0] != mean_imag.shape[0]) ):
#    print 'Bootstrapped operators do not aggree with expected operators'
#    exit(0)
#  
#  print qn.shape

  # subduced correlators
  filename = './bootdata/p%1i/%s_p%1i_subduced.npy' % (p, diagram, p)
  data = np.load(filename)
  mean_sub, err_sub = mean_error_print_foreach_row(data)
  
  filename = './bootdata/p%1i/%s_p%1i_subduced_qn.npy' % \
                                                                 (p, diagram, p)
  qn_sub = np.load(filename)
  if ( (qn_sub.shape[0] != mean_sub.shape[0]) ):
    print 'Bootstrapped operators do not aggree with expected operators'
    exit(0)
  print qn_sub.shape

   # subduced correlators + average over \vec{k1} and \vec{k2}
  filename = './bootdata/p%1i/%s_p%1i_subduced_avg_vecks.npy' % (p, diagram, p)
  data = np.load(filename)
  mean_sub_vecks, err_sub_vecks = mean_error_print(data)
  
  filename = './bootdata/p%1i/%s_p%1i_subduced_avg_vecks_qn.npy' % \
                                                                 (p, diagram, p)
  qn_sub_vecks = np.load(filename)
  if ( (qn_sub_vecks.shape[0] != mean_sub_vecks.shape[0]) ):
    print 'Bootstrapped operators do not aggree with expected operators'
    exit(0)
  print qn_sub_vecks.shape

  # subduced correlators + average over \vec{k1}, \vec{k2} and \mu
  filename = './bootdata/p%1i/%s_p%1i_subduced_avg_rows.npy' % (p, diagram, p)
  data = np.load(filename)
  mean_sub_rows, err_sub_rows = mean_error_print(data)
  
  filename = './bootdata/p%1i/%s_p%1i_subduced_avg_rows_qn.npy' % \
                                                                 (p, diagram, p)
  qn_sub_rows = np.load(filename)
  if ( (qn_sub_rows.shape[0] != mean_sub_rows.shape[0]) ):
    print 'Bootstrapped operators do not aggree with expected operators'
    exit(0)
  print qn_sub_rows.shape

  ################################################################################
  # pick a cmap and get the colors. cool (blueish) for real, autumn (redish) for
  # imaginary correlators
  
  #TODO: put that into function calls
#  cmap_real = plt.cm.cool(np.asarray(range(0,qn.shape[0]))*256/(qn.shape[0]-1))
#  cmap_imag= plt.cm.autumn(np.asarray(range(0,qn.shape[0]))*256/(qn.shape[0]-1))
  


################################################################################
# plotting #####################################################################
################################################################################

  utils.ensure_dir('./plots')

#  plot_path = './plots/Correlators_single_p%1i.pdf' % p
#  pdfplot = PdfPages(plot_path)
#  plot_single(mean_real, err_real, mean_imag, err_imag, qn, pdfplot)
#  pdfplot.close()

  plot_path = './plots/%s_vecks_p%1i.pdf' % (diagram, p)
  pdfplot = PdfPages(plot_path)
  plot_vecks(mean_sub, err_sub, qn_sub, mean_sub_vecks, err_sub_vecks, pdfplot)
  pdfplot.close()

  plot_path = './plots/%s_rows_p%1i.pdf' % (diagram, p)
  pdfplot = PdfPages(plot_path)
  plot_rows(mean_sub_vecks, err_sub_vecks, qn_sub_vecks, mean_sub_rows, err_sub_rows, pdfplot)
  pdfplot.close()

  plot_path = './plots/%s_abs_p%1i.pdf' % (diagram, p)
  pdfplot = PdfPages(plot_path)
  plot_abs(mean_sub, err_sub, qn_sub, mean_sub_vecks, err_sub_vecks, pdfplot)
  pdfplot.close()

#  plot_path = './plots/Correlators_grouped_p%1i.pdf' % p
#  pdfplot = PdfPages(plot_path)
#  avg = plot_grouped(mean_real, err_real, mean_imag, err_imag, qn, \
#                                        gamma, pdfplot, False)
#  pdfplot.close()

#  plot_path = './plots/Correlators_grouped_2_p%1i.pdf' % p
#  pdfplot = PdfPages(plot_path)
#  avg = plot_grouped_2(mean_real, err_real, mean_imag, err_imag, qn, \
#                                        gamma, pdfplot, False)
#  pdfplot.close()

