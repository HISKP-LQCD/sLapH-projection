import matplotlib
#matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.axes as ax

import itertools as it
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import utils

# TODO: Symmetrization and Antisymmetrization. Take negative eigenvalues under 
# time reversal into account
def bootstrap(X, bootstrapsize):
  """
  bootstrapping
  """
  np.random.seed(1227)
  boot = np.empty(bootstrapsize, dtype=float)
  # writing the mean value in the first sample
  boot[0] = np.mean(X)
  # doing all other samples
  for i in range(1, bootstrapsize):
    rnd = np.random.random_integers(0, high=len(X)-1, size=len(X))
    boot_dummy = 0.0 
    for j in range(0, len(X)):
      boot_dummy += X[rnd[j]]  
    boot[i] = boot_dummy/len(X)
  return boot

def mean_and_std(df):
  """
  Mean and standard deviation over all configurations/bootstrap samples

  Parameters
  ----------
  df : pd.DataFrame
      Table with purely real entries (see Notes) and hierarchical columns where
      level 0 is the gauge configuration/bootrstrap sample number and level 1
      is the lattice time

  Returns:
  --------
  pd.DataFrame
      Table with identical indices as `df` The column level 0 is replaced by 
      mean and std while level 1 remains unchanged

  Notes
  -----
  Must be taken for real and imaginary part seperately because that apparently 
  breaks pandas.
  """

  return pd.concat([df.mean(axis=1, level=1), df.std(axis=1, level=1)], axis=1, keys=['mean', 'std'])

def plot_gevp(gevp_data, pdfplot):

  gevp_data = mean_and_std(gevp_data)

  for gevp_el_name, gevp_el_data in gevp_data.iterrows():

    plt.title(r'Gevp Element ${} - {}$'.format(gevp_el_name[0], gevp_el_name[1]))
    plt.xlabel(r'$t/a$', fontsize=12)
    plt.ylabel(r'$C(t/a)$', fontsize=12)

#    print gevp_el_name
#    print series
#    series.index.levels[1]
    T = gevp_el_data.index.levels[1].astype(int)
    mean = gevp_el_data['mean'].values
    std = gevp_el_data['std'].values

    plt.errorbar(T, mean, std, fmt='o', color='black', markersize=3, \
                 capsize=3, capthick=0.75, elinewidth=0.75, \
                                       markeredgecolor='black', linewidth='0.0')

#    plt.legend(numpoints=1, loc=1, fontsize=6)
    pdfplot.savefig()
    plt.clf()
  return 

## does not make sence for CMF C2 because there i just one momentum
#def plot_sep_rows_sep_mom()

def plot_sep_rows_sum_mom(subduced_data, diagram, pdfplot):

  symbol = ['v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', '8']

  subduced_data = mean_and_std(subduced_data)
  # To plot all rows in one plot, transpose row of irrep from index to columns
  rows = subduced_data.index.levels[2].astype(int)
  subduced_data = pd.concat([subduced_data.xs(r, level=2) for r in rows], axis=1, keys=rows)

#  for gevp_el_name, gevp_el_data in subduced_data.iterrows():
  # iterrows() returns a tuple (index, series) where index is a tuple with
  # strings describing the operators for source and sink and series is the 
  # data after mean and std where calculated
  for gevp_el_name, gevp_el in subduced_data.iterrows():

    # prepare plot
    print 'plotting ...'
    plt.title(r'Gevp Element ${} - {}$'.format(gevp_el_name[0], gevp_el_name[1]))
    plt.xlabel(r'$t/a$', fontsize=12)
    plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)

    # prepare data to plot
    rows = gevp_el.index.levels[0]
    for counter, mu in enumerate(rows):

      T = np.array(gevp_el[mu].index.levels[1], dtype=int)
      mean = gevp_el[mu]['mean'].values
      std = gevp_el[mu]['std'].values


      # prepare parameters for plot design
      if len(rows) == 1:
        cmap_brg=['r']
      else:
        cmap_brg = plt.cm.brg(np.asarray(range(len(rows))) * 256/(len(rows)-1))
      shift = 1./3/len(rows)
      label = r'$\mu = %i$' % (counter)

      # plot
      plt.errorbar(T+shift*counter, mean, std, 
                       fmt=symbol[counter%len(symbol)], color=cmap_brg[counter], \
                       label=label, markersize=3, capsize=3, capthick=0.5, \
                       elinewidth=0.5, markeredgecolor=cmap_brg[counter], \
                                                                  linewidth='0.0')


    # clean up for next plot
    plt.legend(numpoints=1, loc='best', fontsize=6)
    pdfplot.savefig()
    plt.clf()

def plot_avg_rows_sep_mom(subduced_data, diagram, pdfplot):

  symbol = ['v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', '8']

  subduced_data = mean_and_std(subduced_data)


  gevp_index = list(set([(g[0],g[1]) for g in subduced_data.index.values]))
  for gevp_el_name in gevp_index:

    # prepare plot
    print 'plotting ', gevp_el_name[0], ' - ', gevp_el_name[1]
    plt.title(r'Gevp Element ${} - {}$'.format(gevp_el_name[0], gevp_el_name[1]))
    plt.xlabel(r'$t/a$', fontsize=12)
    plt.ylabel(r'$%s(t/a)$' % diagram, fontsize=12)

    # prepare data to plot
    gevp_el = subduced_data.xs(gevp_el_name, level=[0,1])
    momenta = gevp_el.index.values

    # iterrows() returns a tuple (index, series) where index is a tuple with
    # strings describing the operators for source and sink and series is the 
    # data after mean and std where calculated
    for counter, (p, gevp_el_p) in enumerate(gevp_el.iterrows()):

      T = gevp_el_p.index.levels[1].values
      mean = gevp_el_p['mean'].values
      std = gevp_el_p['std'].values

      # prepare parameters for plot design
      if len(momenta) == 1:
        cmap_brg=['r']
      else:
        cmap_brg = plt.cm.brg(np.asarray(range(len(momenta))) * 256/(len(momenta)-1))
      shift = 1./3/len(momenta)
      p_so = p[0]
      p_si = p[1]
      label = r'$p_{so} = %s - p_{si} = %s$' % (p_so, p_si)

      # plot
      plt.errorbar(T+shift*counter, mean, std, 
                       fmt=symbol[counter%len(symbol)], color=cmap_brg[counter], \
                       label=label, markersize=3, capsize=3, capthick=0.5, \
                       elinewidth=0.5, markeredgecolor=cmap_brg[counter], \
                                                                  linewidth='0.0')


    # clean up for next plot
    plt.legend(numpoints=1, loc='best', fontsize=6)
    pdfplot.savefig()
    plt.clf()


p_cm = 0
irrep = 'T1'

for diagram in ['C2', 'C3', 'C4']:
  path = './readdata/%s_p%1i_%s.h5' % (diagram, p_cm, irrep)
  subduced_data = utils.read_hdf5_correlators(path, False)
  # discard imaginary part (noise)
  subduced_data = subduced_data.apply(np.real)
  # sum over all gamma structures to get the full Dirac operator transforming 
  # like a row of the desired irrep
  subduced_data = subduced_data.sum(level=[0,1,2,3,5])
  # sum over equivalent momenta
  subduced_data = subduced_data.sum(level=[0,1,2])
  
  utils.ensure_dir('./plots')
  plot_path = './plots/%s_sep_rows_sum_mom_p%1i_%s.pdf' % (diagram, p_cm, irrep)
  pdfplot = PdfPages(plot_path)
  plot_sep_rows_sum_mom(subduced_data, diagram, pdfplot)
  pdfplot.close()


  path = './readdata/%s_p%1i_%s.h5' % (diagram, p_cm, irrep)
  subduced_data = utils.read_hdf5_correlators(path, False)
  # discard imaginary part (noise)
  subduced_data = subduced_data.apply(np.real)
  # sum over all gamma structures to get the full Dirac operator transforming 
  # like a row of the desired irrep
  subduced_data = subduced_data.sum(level=[0,1,2,3,5])
  # average over rows
  subduced_data = subduced_data.mean(level=[0,1,3,4])
  
  utils.ensure_dir('./plots')
  plot_path = './plots/%s_avg_rows_sep_mom_p%1i_%s.pdf' % (diagram, p_cm, irrep)
  pdfplot = PdfPages(plot_path)
  plot_avg_rows_sep_mom(subduced_data, diagram, pdfplot)
  pdfplot.close()

path = './readdata/Gevp_p%1i_%s.h5' % (p_cm, irrep)
gevp_data = utils.read_hdf5_correlators(path, False)

utils.ensure_dir('./plots')
plot_path = './plots/Gevp_p%1i_%s.pdf' % (p_cm, irrep)
pdfplot = PdfPages(plot_path)
plot_gevp(gevp_data, pdfplot)
pdfplot.close()

