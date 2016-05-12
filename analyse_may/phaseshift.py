#!/usr/bin/python

import sys
import numpy as np

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab

import boot
import histogram as hist
import plot as plt
import IOcontraction

import phaseshift as calc

T = 48 # temporal extend of the lattice
L = 24
pvalue_cut = 0.0 # minimal p-value
min_fitrange = 4 # minimal fit range
write = 0

################################################################################
plot_path = './plots/Ecm_histogramms.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot_Ecm = PdfPages(plot_path)

plot_path = './plots/q2_histogramms.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot_q2 = PdfPages(plot_path)

plot_path = './plots/ps_histogramms.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot_ps = PdfPages(plot_path)

plot_path = './plots/tan_ps_histogramms.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot_tan_ps = PdfPages(plot_path)

plot_path = './plots/sin_ps_histogramms.pdf'
IOcontraction.ensure_dir(plot_path)
pdfplot_sin_ps = PdfPages(plot_path)


################################################################################

#sys.stdout = open('./phaseshift.out', 'w')

#read 2pt masses for cms.
filename = 'bootdata/C2_massfit_p0_median.npy'
C2_mass_median= np.load(filename)

for p in range(1,2):

  for ev in range(0,1):
    filename = 'bootdata/E%d_4pt_massfit_p%d.npy' % (ev, p)
    C4 = np.load(filename)
    filename = 'bootdata/E%d_4pt_massfit_params_p%d.npy' % (ev, p)
    C4_params = np.load(filename)
  
    # choosing combinations with wanted p-values and fit ranges
    ############################################################################
    C4_mass, C4_mass_params = \
             hist.choose_pvalue_cut(C4, C4_params, pvalue_cut, min_fitrange)
  
    ############################################################################
    # calculate Ecm and q2
    gamma, Ecm = calc.calc_Ecm(C4_mass, p, L)

    print 'center-of-mass energy of eigenvalue E%d for p=%d:' % (ev, p)
    Ecm_median, Ecm_weight, Ecm_std, Ecm_sys_lo, Ecm_sys_hi = \
                                           hist.compute_syst_err(Ecm, C4_params)
  

    #TODO: Check whether the same fitintervals must be used for 2pt function
    q2 = calc.calc_q2(Ecm, C2_mass_median, L)

    if write:
      print 'q2 of eigenvalue E%d for p=%d:' % (ev, p)
    q2_median, q2_weight, q2_std, q2_sys_lo, q2_sys_hi = \
                                 hist.compute_syst_err(q2, C4_params, write)
    ###########################################################################
    # calculate phaseshift ps 
    ps, tan_ps, sin_ps = calc.calc_ps(q2, gamma, p) 
    write = 1
    if write:
      print 'phase shift of eigenvalue E%d for p=%d:' % (ev, p)
    ps_median, ps_weight, ps_std, ps_sys_lo, ps_sys_hi = \
                                     hist.compute_syst_err(ps, C4_params, write)
    write = 0
    if write:
      print 'tan(phase shift) of eigenvalue E%d for p=%d:' % (ev, p)
    tan_ps_median, tan_ps_weight, tan_ps_std, tan_ps_sys_lo, tan_ps_sys_hi = \
                                 hist.compute_syst_err(tan_ps, C4_params, write)
    if write:
      print 'sin(phase shift) of eigenvalue E%d for p=%d:' % (ev, p)
    sin_ps_median, sin_ps_weight, sin_ps_std, sin_ps_sys_lo, sin_ps_sys_hi = \
                                 hist.compute_syst_err(sin_ps, C4_params, write)

    ###########################################################################
    # plot histograms
    label = [r'$E_{CM}$', r'weighted distribution of $E_{CM}$', \
             r'p=%d, EV%d, $E_{CM} = %.4f \pm %.4f + %.4f - %.4f$ ' % \
                                      (p, ev, Ecm_median[0], Ecm_std, \
                                       Ecm_sys_hi, Ecm_sys_lo)]
    plt.hist(Ecm, Ecm_median, Ecm_weight, Ecm_std, Ecm_sys_lo, Ecm_sys_hi, \
             label, pdfplot_Ecm)

    label = [r'$q^2$', r'weighted distribution of $q^2$', \
             r'p=%d, EV%d, $q^2 = %.4f \pm %.4f + %.4f - %.4f$ ' % \
                                      (p, ev, q2_median[0], q2_std, \
                                       q2_sys_hi, q2_sys_lo)]
    plt.hist(q2, q2_median, q2_weight, q2_std, q2_sys_lo, q2_sys_hi, \
             label, pdfplot_q2)

    label = [r'$\delta$', r'weighted distribution of $\delta$', \
             r'p=%d, EV%d, $\delta = %.4f \pm %.4f + %.4f - %.4f$ ' % \
                                      (p, ev, ps_median[0], ps_std, \
                                       ps_sys_hi, ps_sys_lo)]
    plt.hist(ps, ps_median, ps_weight, ps_std, ps_sys_lo, ps_sys_hi, \
             label, pdfplot_ps)

    label = [r'$\tan(\delta)$', r'weighted distribution of \tan(\delta)', \
             r'p=%d, EV%d, $\tan(\delta) = %.4f \pm %.4f + %.4f - %.4f$ ' % \
                                      (p, ev, tan_ps_median[0], tan_ps_std, \
                                       tan_ps_sys_hi, tan_ps_sys_lo)]
    plt.hist(tan_ps, tan_ps_median, tan_ps_weight, tan_ps_std, \
             tan_ps_sys_lo, tan_ps_sys_hi, label, pdfplot_tan_ps)

    label = [r'$\sin(\delta)$', r'weighted distribution of \sin(\delta)', \
             r'p=%d, EV%d, $\sin(\delta) = %.4f \pm %.4f + %.4f - %.4f$ ' % \
                                      (p, ev, sin_ps_median[0], sin_ps_std, \
                                       sin_ps_sys_hi, sin_ps_sys_lo)]
    plt.hist(sin_ps, sin_ps_median, sin_ps_weight, sin_ps_std, \
             sin_ps_sys_lo, sin_ps_sys_hi, label, pdfplot_sin_ps)

    ###########################################################################
    # save data and results of error calculation
    params = np.array([Ecm_median, Ecm_weight, Ecm_std, Ecm_sys_lo, Ecm_sys_hi])
    path = 'bootdata/E%d_cm_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, Ecm)
    path = 'bootdata/E%d_cm_params_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, params)

    params = np.array([q2_median, q2_weight, q2_std, q2_sys_lo, q2_sys_hi])
    path = 'bootdata/E%d_q2_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, q2)
    path = 'bootdata/E%d_q2_params_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, params)

    params = np.array([ps_median, ps_weight, ps_std, ps_sys_lo, ps_sys_hi])
    path = 'bootdata/E%d_ps_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, ps)
    path = 'bootdata/E%d_ps_params_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, params)

    params = np.array([tan_ps_median, tan_ps_weight, tan_ps_std, \
                       tan_ps_sys_lo, tan_ps_sys_hi])
    path = 'bootdata/E%d_tan_ps_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, tan_ps)
    path = 'bootdata/E%d_tan_ps_params_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, params)
   
    params = np.array([sin_ps_median, sin_ps_weight, sin_ps_std, \
                       sin_ps_sys_lo, sin_ps_sys_hi])
    path = 'bootdata/E%d_sin_ps_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, sin_ps)
    path = 'bootdata/E%d_sin_ps_params_p%d' % (ev, p)
    IOcontraction.ensure_dir(path)
    np.save(path, params)

    ###########################################################################

pdfplot_Ecm.close()
pdfplot_q2.close()
pdfplot_ps.close()
pdfplot_tan_ps.close()
pdfplot_sin_ps.close()


