#!/usr/bin/python

import math
import numpy as np
import scipy.optimize
import scipy.stats
from itertools import chain

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab

import histogram as hist
import plot as plt

import IOcontraction

#import sys
#sys.stdout = open('./dispersion_relation_pi.out', 'w')

# some global variables
################################################################################
p = 0 # momenta to analyse
L = 24 # lattice extend
pvalue_cut = 0.0 # minimal p-value
min_fitrange = 4 # minimal fit range


# writing the 4pt pion mass out and creating a histogram with all data
################################################################################
plot_path = './plots/EV_histogramms.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot = PdfPages(plot_path)

################################################################################
################################################################################
################################################################################
# (4pt) reading data extracted from direct fit
for p in range(0,5):
  for ev in range(0,2):
    filename = 'bootdata/E%d_4pt_massfit_p%d.npy' % (ev, p)
    C4_read = np.load(filename)
    filename = 'bootdata/E%d_4pt_massfit_params_p%d.npy' % (ev, p)
    C4_params = np.load(filename)
  
    # choosing combinations with wanted p-values and fit ranges
    ############################################################################
    C4_mass, C4_mass_params = \
            hist.choose_pvalue_cut(C4_read, C4_params, pvalue_cut, min_fitrange)
  
    C4_median, C4_weight, C4_std, C4_sys_lo, C4_sys_hi = \
                         hist.compute_syst_err(C4_mass, C4_mass_params, write=0)

    label = [r'$E_{eff}$', r'weighted distribution of E_{eff}', \
             r'p=%d, EV%d, $E_{eff} = %.4f \pm %.4f + %.4f - %.4f$ ' % \
                                      (p, ev, C4_median[0], C4_std, \
                                       C4_sys_hi, C4_sys_lo)]
    plt.hist(C4_mass, C4_median, C4_weight, C4_std, C4_sys_lo, C4_sys_hi, \
             label, pdfplot)

pdfplot.close()


