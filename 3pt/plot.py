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
gamma_50i = [13, 14, 15, \
             ['\gamma_5\gamma_0\gamma_1', '\gamma_5\gamma_0\gamma_2', \
              '\gamma_5\gamma_0\gamma_3', '\gamma_5\gamma_0\gamma_i']]
gamma_5 = [5, ['\gamma_5']]
gamma_for_filenames = {'\gamma_i' : 'gi', \
                       '\gamma_0\gamma_i' : 'g0gi', \
                       '\gamma_5\gamma_0\gamma_i' : 'g5g0gi'}

gammas = [gamma_i, gamma_0i, gamma_50i]
#gammas = [gamma_i, gamma_50i]

# list of all filled symbols in matplotlib except 'o'
symbol = ['v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', '8']

diagram = 'C3+'

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
    for k,k1k2 in enumerate(irrep):
      for g, gamma in enumerate(k1k2):
        for r,row in enumerate(gamma):
          mean[i,k,g,r] = np.mean(row, axis=-1)
          err[i,k,g,r]  = np.std(row, axis=-1)
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

def read_ensemble(p, name, foreach_row=False):

  # subduced correlators
  filename = './bootdata/p%1i/%s.npy' % (p, name)
  data = np.load(filename)
  if not foreach_row:
    mean, err = mean_error_print(data)
  else:
    mean, err = mean_error_print_foreach_row(data)
  
  filename = './bootdata/p%1i/%s_qn.npy' % (p, name)
  qn = np.load(filename)
  if ( (qn.shape[0] != mean.shape[0]) ):
    print 'Bootstrapped operators %s do not aggree with expected operators' % \
                                                                            name
    exit(0)
  print qn.shape

  return mean, err, qn

################################################################################
# plotting functions ###########################################################
################################################################################

################################################################################
# plots all correlators into  one plot for each operator. Real and imaginary 
# part into same plot with their own axes
def plot_single(mean_real, err_real, mean_imag, err_imag, qn, pdfplot):

  # pick a cmap and get the colors. cool (blueish) for real, autumn (redish) for
  # imaginary correlators
  cmap_real = plt.cm.cool(np.asarray(range(0,qn.shape[0]))*256/(qn.shape[0]-1))
  cmap_imag= plt.cm.autumn(np.asarray(range(0,qn.shape[0]))*256/(qn.shape[0]-1))

  for op in range(0, qn.shape[0]):
    if verbose:
      print 'plot_single op %i' % op
    plt.title(r'$p_{so} = [(%i,%i,%i),(%i,%i,%i)] \ d_{so} = ' \
               '[(%i,%i,%i),(%i,%i,%i)] \ \gamma_{so} = [%2i,%2i] ' \
               '\quad p_{si} = (%i,%i,%i) \ d_{si} = (%i,%i,%i) \ ' \
               '\gamma_{si} = %2i$' % \
               (qn[op][0][0], qn[op][0][1], qn[op][0][2], \
                                     qn[op][6][0], qn[op][6][1], qn[op][6][2], \
                qn[op][1][0], qn[op][1][1], qn[op][1][2], \
                qn[op][7][0], qn[op][7][1], qn[op][7][2], qn[op][2], qn[op][8], \
                qn[op][3][0], qn[op][3][1], qn[op][3][2], \
                         qn[op][4][0], qn[op][4][1], qn[op][4][2], qn[op][5]), \
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
    real = par1.errorbar(range(0, mean_real.shape[-1]), mean_real[op], \
                         err_real[op], fmt=symbol[op%len(symbol)], \
                         color=cmap_real[op], label='real', markersize=3, \
                         capsize=3, capthick=0.5, elinewidth=0.5, \
                                 markeredgecolor=cmap_real[op], linewidth='0.0')
    imag = par2.errorbar(range(0, mean_imag.shape[-1]), mean_imag[op], \
                         err_imag[op], fmt=symbol[op%len(symbol)], \
                         color=cmap_imag[op], label='imag', markersize=3, \
                         capsize=3, capthick=0.5, elinewidth=0.5, \
                                 markeredgecolor=cmap_imag[op], linewidth='0.0')
  
    pdfplot.savefig()
    plt.clf()

  print ' '
  return

################################################################################
# plot all combinations of momenta at source/ sink subducing into the same
# \Lambda, [|\vec{k1}|, |\vec{k2}|], \mu, \vec{P}
def plot_vecks(mean_sin, err_sin, qn_sin, mean_avg, err_avg, gammas, pdfplot):

#  ax = plt.subplot(111)

  # TODO: include loop over gamma structure and create additional 
  # array-dimension with just \gamma_5 as entry
  for i, irrep in enumerate(qn_sin):
    for k, gevp_row in enumerate(irrep):
      for g, gevp_col in enumerate(gevp_row):
        for r, row in enumerate(gevp_col):
          print 'plot row %i of irrep %s, [%i,%i] -> %i' % (r, row[0,-1], \
                 row[0,-5][0], row[0,-5][1], p)
  
          cmap_brg = plt.cm.brg(np.asarray(range(0, row.shape[0])) * \
                                           256/(row.shape[0]-1))
          if verbose:
            print row.shape[0] 
          shift = 1./3/row.shape[0]
          for op in range(0, row.shape[0]):
  
          #TODO: title
            # set plot title, labels etc.
            plt.title(r'$%s%s$ - $%s$ Operators ' \
                      r'subduced into $p = %i$, $[%i,%i] \ \to \ [%i$] ' \
                      r'under $\Lambda = %s$ $\mu = %i$' % \
                        (gamma_5[-1][-1], gamma_5[-1][-1], \
                         gammas[g][-1][-1], p, row[op][-5][0], row[op][-5][1], \
                         p, row[op][-1], r+1),\
                      fontsize=12)
            plt.xlabel(r'$t/a$', fontsize=12)
            plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)
    
  #          if abs(mean_sin[i,k,r][op,0]) >= 0.05*abs(np.max(mean_sin[i,k,r][:,0])):
            label = r'$[(%2i,%2i,%2i), (%2i,%2i,%2i)] \ \to \ ' \
                    r'[(%2i,%2i,%2i)]$' % \
                      (row[op][0][0], row[op][0][1], row[op][0][2], \
                       row[op][1][0], row[op][1][1], row[op][1][2], \
                       row[op][2][0], row[op][2][1], row[op][2][2])
  #          else:
  #            label = '_nolegend_'
            
            # prepare data for plotting
            # TODO: put that in subduction
            mean = mean_sin[i,k,g,r][op,]
            err = err_sin[i,k,g,r][op,]
                  
  #            # Shrink current axis by 20%
  #            box = ax.get_position()
  #            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  #
  #            # Put a legend to the right of the current axis
  #            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  
  #          plt.yscale('log')
            plt.errorbar(np.asarray(range(0, mean.shape[-1]))+op*shift, mean, err, \
                         fmt=symbol[op%len(symbol)], color=cmap_brg[op], \
                         label=label, markersize=3, capsize=3, capthick=0.5, \
                         elinewidth=0.5, markeredgecolor=cmap_brg[op], \
                                                                    linewidth='0.0')
  
          mean = mean_avg[i,k,g,r,]
          err = err_avg[i,k,g,r,]
          plt.errorbar(np.asarray(range(0, mean.shape[-1]))+op*shift, mean, err, \
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
def plot_rows(mean_sin, err_sin, qn_sin, mean_avg, err_avg, gammas, pdfplot, plot_mean=True):

  # TODO: include loop over gamma structure and create additional 
  # array-dimension with just \gamma_5 as entry
  for i, irrep in enumerate(qn_sin):
    for k, gevp_row in enumerate(irrep):
      for g, gevp_col in enumerate(gevp_row):
        print 'plot irrep %s, [%i,%i] -> %i' % (gevp_col[0,-1], \
               gevp_col[0,-5][0], gevp_col[0,-5][1], p)
  
        cmap_brg = plt.cm.brg(np.asarray(range(0, gevp_col.shape[0])) * \
                                         256/(gevp_col.shape[0]-1))
        if verbose:
          print gevp_col.shape[0] 
        shift = 1./3/gevp_col.shape[0]
        for op in range(gevp_col.shape[0]):
  
          #TODO: title
          # set plot title, labels etc.
          plt.title(r'$%s%s$ - $%s$ Operators ' \
                    r'subduced into $p = %i$, $[%i,%i] \ \to \ [%i]$ ' \
                    r'under $\Lambda = %s$' % \
                      (gamma_5[-1][-1], gamma_5[-1][-1], gammas[g][-1][-1], p, \
                       gevp_col[op][-5][0], gevp_col[op][-5][1], \
                       p, gevp_col[op][-1]),\
                    fontsize=12)
          plt.xlabel(r'$t/a$', fontsize=12)
          plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)
  
   
  #        if abs(mean_sin[i,k,op,0]) >= abs(err_avg[i,k,0]):
          label = r'$\mu = %i$' % (op+1)
  #        else:
  #          label = '_nolegend_'
          
          # prepare data for plotting
          # TODO: put that in subduction
          mean = mean_sin[i,k,g,op]
          err = err_sin[i,k,g,op]

               
          plt.yscale('log')
          plt.errorbar(np.asarray(range(0, mean.shape[-1]))+op*shift, mean, err, \
                       fmt=symbol[op%len(symbol)], color=cmap_brg[op], \
                       label=label, markersize=3, capsize=3, capthick=0.5, \
                       elinewidth=0.5, markeredgecolor=cmap_brg[op], \
                                                                  linewidth='0.0')
  
        if plot_mean:
          mean = mean_avg[i,k,g]
          err = err_avg[i,k,g]
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
def plot_abs(mean_sin, err_sin, qn_sin, mean_avg, err_avg, gammas, pdfplot):

#  ax = plt.subplot(111)

  # TODO: include loop over gamma structure and create additional 
  # array-dimension with just \gamma_5 as entry
  for i, irrep in enumerate(qn_sin):
    for k, gevp_row in enumerate(irrep):
      for g, gevp_col in enumerate(gevp_row):
        for r, row in enumerate(gevp_col):
          print 'plot row %i of irrep %s, [%i,%i] -> %i' % (r, row[0,-1], \
                 row[0,-5][0], row[0,-5][1], p)
  
          cmap_brg = plt.cm.brg(np.asarray(range(0, row.shape[0])) * \
                                           256/(row.shape[0]-1))
          if verbose:
            print row.shape[0] 
          shift = 1./3/row.shape[0]
          for op in range(0, row.shape[0]):

             # set plot title, labels etc.
            plt.title(r'$%s%s$ - $%s$ Operators ' \
                      r'subduced into $p = %i$, $[%i,%i] \ \to \ [%i$] ' \
                      r'under $\Lambda = %s$ $\mu = %i$' % \
                        (gamma_5[-1][-1], gamma_5[-1][-1], \
                         gammas[g][-1][-1], p, row[op][-5][0], row[op][-5][1], \
                         p, row[op][-1], r+1),\
                      fontsize=12)
            plt.xlabel(r'$t/a$', fontsize=12)
            plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)
    
  #          if abs(mean_sin[i,k,r][op,0]) >= 0.05*abs(np.max(mean_sin[i,k,r][:,0])):
            label = r'$[(%2i,%2i,%2i), (%2i,%2i,%2i)] \ \to \ ' \
                    r'[(%2i,%2i,%2i)]$' % \
                      (row[op][0][0], row[op][0][1], row[op][0][2], \
                       row[op][2][0], row[op][2][1], row[op][2][2], \
                       row[op][1][0], row[op][1][1], row[op][1][2])
  #          else:
  #            label = '_nolegend_'
            
            # prepare data for plotting
            # TODO: takewhile breaks one iteration to early
            mean = it.takewhile(lambda (x,y): x/y > 0, \
                                it.izip(mean_sin[i,k,g,r][op,1:22], mean_sin[i,k,g,r][op,2:23]))
            mean = np.asarray(list(abs(m[1]) for m in mean))
            mean = np.insert(mean, 0, abs(mean_sin[i,k,g,r][op,1]))
            err = err_sin[i,k,g,r][op,1:(mean.shape[0]+1)]
                  
  #            # Shrink current axis by 20%
  #            box = ax.get_position()
  #            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  #
  #            # Put a legend to the right of the current axis
  #            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  
            plt.yscale('log')
            plt.errorbar(np.asarray(range(1, mean.shape[-1]+1))+op*shift, mean, err, \
                         fmt=symbol[op%len(symbol)], color=cmap_brg[op], \
                         label=label, markersize=3, capsize=3, capthick=0.5, \
                         elinewidth=0.5, markeredgecolor=cmap_brg[op], \
                                                                    linewidth='0.0')
  
  #        mean = mean_avg[i,k,r,:23]
  #        err = err_avg[i,k,r,:23]
  #        plt.errorbar(np.asarray(range(0, 23))+op*shift, mean, err, \
  #                     fmt='o', color='black', \
  #                     label='average', markersize=3, capsize=3, capthick=0.5, \
  #                     elinewidth=0.5, markeredgecolor='black', \
  #                                                                linewidth='0.0')
      
#          plt.legend(numpoints=1, loc='best', fontsize=6).get_frame().set_alpha(0.5)
          pdfplot.savefig()
          plt.clf()

  print ' '
  return

################################################################################
# plot all combinations of momenta at source/ sink subducing into the same
# \Lambda, [|\vec{k1}|, |\vec{k2}|], \mu, \vec{P}
def plot_signal_to_noise(mean_sin, err_sin, qn_sin, mean_avg, err_avg, gammas, pdfplot):

#  ax = plt.subplot(111)

  for i, irrep in enumerate(qn_sin):
    for k, gevp_row in enumerate(irrep):
      for g, gevp_col in enumerate(gevp_row):
        for r, row in enumerate(gevp_col):
          print 'plot row %i of irrep %s, [%i,%i] -> %i' % (r, row[0,-1], \
                 row[0,-5][0], row[0,-5][1], p)
  
          cmap_brg = plt.cm.brg(np.asarray(range(0, row.shape[0])) * \
                                           256/(row.shape[0]-1))
          if verbose:
            print row.shape[0] 
          shift = 1./3/row.shape[0]
          for op in range(0, row.shape[0]):
  
          #TODO: title
            # set plot title, labels etc.
            plt.title(r'$%s%s$ - $%s$ Operators ' \
                      r'subduced into $p = %i$, $[%i,%i] \ \to \ [%i$] ' \
                      r'under $\Lambda = %s$ $\mu = %i$' % \
                        (gamma_5[-1][-1], gamma_5[-1][-1], \
                         gammas[g][-1][-1], p, row[op][-5][0], row[op][-5][1], \
                         p, row[op][-1], r+1),\
                      fontsize=12)
            plt.xlabel(r'$t/a$', fontsize=12)
            plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)
            plt.yscale('log')
#            plt.ylim((-2,2))
    
  #          if abs(mean_sin[i,k,r][op,0]) >= 0.05*abs(np.max(mean_sin[i,k,r][:,0])):
            label = r'$[(%2i,%2i,%2i), (%2i,%2i,%2i)] \ \to \ ' \
                    r'[(%2i,%2i,%2i)]$' % \
                      (row[op][0][0], row[op][0][1], row[op][0][2], \
                       row[op][1][0], row[op][1][1], row[op][1][2], \
                       row[op][2][0], row[op][2][1], row[op][2][2])
  #          else:
  #            label = '_nolegend_'
            
            # prepare data for plotting
            # TODO: put that in subduction
            mean = mean_sin[i,k,g,r][op,]
            err = err_sin[i,k,g,r][op,]
                  
  #            # Shrink current axis by 20%
  #            box = ax.get_position()
  #            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  #
  #            # Put a legend to the right of the current axis
  #            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  
            plt.plot(np.asarray(range(0, mean.shape[-1]))+op*shift, np.abs(err/mean), \
                     marker=symbol[op%len(symbol)], color=cmap_brg[op], \
                     label=label, markersize=3, markeredgecolor=cmap_brg[op], \
                                                                linewidth='0.0')
  
          mean = mean_avg[i,k,g,r,]
          err = err_avg[i,k,g,r,]
          plt.plot(np.asarray(range(0, mean.shape[-1]))+op*shift, np.abs(err/mean), \
                   marker='o', color='black', label='average', markersize=3, \
                   markeredgecolor='black', linewidth='0.0')
          plt.plot([0,mean.shape[-1]], [1,1], color='black', linestyle='--')
#          plt.plot([0,mean.shape[-1]], [-1,-1], color='black', linestyle='--')
      
          plt.legend(numpoints=1, loc='best', fontsize=6).get_frame().set_alpha(0.5)
          pdfplot.savefig()
          plt.clf()
  
  print ' '
  return

################################################################################
# read data

for p in [0]:


  diagram = 'C3+'

  # bootstrapped correlators
  print 'reading bootstrapped correlators'
  name = '%s_p%1i_real' % (diagram, p)
  mean_real, err_real, qn = read_ensemble(p, name)

  name = '%s_p%1i_imag' % (diagram, p)
  mean_imag, err_imag, qn = read_ensemble(p, name)

  if (mean_real.shape[0] != mean_imag.shape[0]):
    print 'Real and imaginary part of bootstrapped operators do not aggree'
    exit(0)

  # subduced correlators
  print 'reading subduced correlators'
  name = '%s_p%1i_subduced' % (diagram, p)
  mean_sub, err_sub, qn_sub = read_ensemble(p, name, True)

  # subduced correlators + average over \vec{k1} and \vec{k2}
  print 'reading subduced correlators averaged over three-momenta'
  name = '%s_p%1i_subduced_avg_vecks' % (diagram, p)
  mean_sub_vecks, err_sub_vecks, qn_sub_vecks = read_ensemble(p, name)

  # subduced correlators + average over \vec{k1}, \vec{k2} and \mu
  print 'reading subduced correlators averaged over three-momenta and rows'
  name = '%s_p%1i_subduced_avg_rows' % (diagram, p)
  mean_sub_rows, err_sub_rows, qn_sub_rows = read_ensemble(p, name)

################################################################################
# plotting #####################################################################
################################################################################

  utils.ensure_dir('./plots')

#  plot_path = './plots/%s_single_p%1i.pdf' % (diagram, p)
#  pdfplot = PdfPages(plot_path)
#  plot_single(mean_real, err_real, mean_imag, err_imag, qn, pdfplot)
#  pdfplot.close()

  plot_path = './plots/%s_vecks_p%1i.pdf' % (diagram, p)
  pdfplot = PdfPages(plot_path)
  plot_vecks(mean_sub, err_sub, qn_sub, mean_sub_vecks, err_sub_vecks, \
                                                                gammas, pdfplot)
  pdfplot.close()

  plot_path = './plots/%s_rows_p%1i.pdf' % (diagram, p)
  pdfplot = PdfPages(plot_path)
  plot_rows(mean_sub_vecks, err_sub_vecks, qn_sub_vecks, mean_sub_rows,  \
                                           err_sub_rows, gammas, pdfplot, False)
  pdfplot.close()

  plot_path = './plots/%s_abs_p%1i.pdf' % (diagram, p)
  pdfplot = PdfPages(plot_path)
  plot_abs(mean_sub, err_sub, qn_sub, mean_sub_vecks, err_sub_vecks, gammas, \
                                                                        pdfplot)
  pdfplot.close()

#  plot_path = './plots/%s_relerr_p%1i.pdf' % (diagram, p)
#  pdfplot = PdfPages(plot_path)
#  plot_signal_to_noise(mean_sub, err_sub, qn_sub, mean_sub_vecks, err_sub_vecks, gammas, pdfplot)
#  pdfplot.close()

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

