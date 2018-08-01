from ast import literal_eval
import ConfigParser
import numpy as np
import collections
import sys
import pandas as pd
from pandas import Series, DataFrame


import src.raw_data as raw_data
import src.utils as utils
import src.wick as wick
import src.projection as projection
import src.subduction as subduction
import src.setup_gevp as setup_gevp
import src.plot as plot
import src.plot_bokeh as bokeh

############################################################################
# Main
# TODO: pull out irrep as outer loop and do not use it in any of the files
# TODO: refactor towards a more objectoriented design
def main(process, flag, sta_cnfg, end_cnfg, del_cnfg, missing_configs, ensemble,
        T, list_of_pcm_sq, p_cutoff, default_list_of_pcm, gamma_input, default_list_of_q,
        default_beta, list_of_diagrams, directories, path_to_sc, path_to_sc_2, outpath,
        plot_p_and_g, plot_pcm_and_mu, plot_avg, plot_experimental, logscale,
        bootstrapsize, continuum_basis_string, verbose):

    for p_cm_sq in list_of_pcm_sq:
        if verbose:
            print 80 * '#'
            print 'p_cm_sq = ', p_cm_sq
    
        ########################################################################
        # read diagrams for correlators contributing to rho
        if flag['read']:
            path = '%s/%s/0_raw-data/' % (outpath, ensemble)
            raw_data.read(path, T, list_of_diagrams, directories, sta_cnfg, end_cnfg, del_cnfg,
                    missing_configs, process, p_cm_sq, p_cutoff, gamma_input, verbose)
    
#        # TODO: Is gamma_input giving the right quantum numbers?
#        if flag_pion:
#            path = '%s/%s/0_raw-data/' % (outpath, ensemble)
#            raw_data.read(path, T, ['C2c'], directories[:1], sta_cnfg, end_cnfg, del_cnfg,
#                    missing_configs, process, p_cm_sq, p_cutoff, gamma_input, verbose)
#    
#            pion_data = utils.read_hdf5_correlators(path + 'C2c_p%1i.h5' % p_cm_sq, 'data')
#            path = '%s/%s/3_gevp-data/' % (outpath, ensemble)
#            filename = 'pi_p%1i.dat' % (p_cm_sq)
#            utils.write_ascii_correlators(
#              path, filename, pion_data.mean(axis=1).apply(np.real), verbose)
    
    
        ########################################################################
        # Setup of list of pcm and irrep to loop over
        path = '%s/%s/0_raw-data/' % (outpath, ensemble)
        filename = '%s_p%1i_qn.h5' % (list_of_diagrams[0], p_cm_sq)
        lookup_qn = utils.read_hdf5_correlators(path + filename, 'qn')
    
        # Angular momentum for particles of interest
        if process == 'rho':
            j = 1
        elif process == 'pipi':
            j = 0
    
        list_of_irreps = projection.get_list_of_irreps(
                lookup_qn['p_{cm}'].unique(), path_to_sc, j)
        correlators = wick.set_lookup_correlators(list_of_diagrams)
    
        for irrep in list_of_irreps:
            print '\t  Subducing into %s' % irrep
    
            ##########################################################################
            # Handling of gevp parameters
            gevp_parameters = ConfigParser.SafeConfigParser(
                    {'list of p_cm' : default_list_of_pcm,
                        'list of q' : default_list_of_q,
                        'beta' : default_beta})
    
            if(gevp_parameters.read('%s/%s/3_gevp-data/p%1d_%s.ini' % 
                    (outpath, ensemble, p_cm_sq, irrep)) == []):
                print 'Did not find gevp parameters for {} in '\
                    '{}/{}/3_gevp-data/p{}_{}.ini. Continuing with default values'.\
                    format(irrep, outpath, ensemble, p_cm_sq, irrep)
                list_of_pcm = default_list_of_pcm
                list_of_q = default_list_of_q
                beta = default_beta
            else:
                list_of_pcm = gevp_parameters.get('gevp parameters', 'list of p_cm')
                list_of_q = gevp_parameters.get('gevp parameters', 'list of q')
                beta = gevp_parameters.get('gevp parameters', 'beta')
    
            # helper function to read all subduced data from disk
            if list_of_pcm == None:
                list_of_pcm = lookup_qn['p_{cm}'].apply(literal_eval).unique()
            else:
                list_of_pcm = [literal_eval(p_cm.strip()) for p_cm in list_of_pcm.split(';')]
            assert all([utils._abs2(p) == p_cm_sq for p in list_of_pcm]), \
                    'list_of_pcm contains momenta with the wrong absolute value squared'
            list_of_pcm = [str(p) for p in list_of_pcm]
    
            if list_of_q != None: 
                list_of_q = [literal_eval(q.strip()) for q in list_of_q.split(';')]
    
            contracted_data_avg = {}
            for correlator, diagrams in correlators.iteritems():
    
                ######################################################################
                #Projection
    
                lattice_operators = projection.project(j, correlator, 
                        continuum_basis_string, gamma_input, list_of_pcm, list_of_q,
                        path_to_sc, path_to_sc_2, verbose)
                lattice_operators = lattice_operators.xs(beta, level=r'\beta')
    
                # Select irrep and discard imaginary part (noise)
                # :warning: multiplicity 1 hardcoded
                lattice_operators = lattice_operators.xs(
                        (irrep, 1), level=('Irrep', 'mult'))
    
                ######################################################################
                # Subduction
    
                subduced_data = {}
                for diagram in diagrams:
    
                    if flag['subduce']:
                        print '\tSubducing data for %s' % diagram
    
                        path = '%s/%s/0_raw-data/' % (outpath, ensemble)
                        filename = '%s_p%1i.h5' % (diagram, p_cm_sq)
                        data = utils.read_hdf5_correlators(path + filename, 'data')
                        filename = '%s_p%1i_qn.h5' % (diagram, p_cm_sq)
                        lookup_qn = utils.read_hdf5_correlators(path + filename, 'qn')
    
                        lookup_corr = subduction.set_lookup_corr(
                          lattice_operators, lookup_qn, verbose)
    
                        subduced_data[diagram] = subduction.project_correlators(data, 
                                lookup_corr)
                        del data
    
                        # write data to disc
                        path = '%s/%s/1_subduced-data/' % (outpath, ensemble)
    
                        filename = '/%s_p%1i_%s.h5' % (diagram, p_cm_sq, irrep)
                        utils.write_hdf5_correlators(
                          path, filename, subduced_data[diagram], 'data', verbose)
    
                        lattice_operators.sort_index().\
                            reset_index(["operator_label_{so}", "operator_label_{si}"],
                                drop=True).\
                            to_csv(path+"lattice-operators_{}_p{}_{}.tsv".format(
                                diagram, p_cm_sq, irrep), sep="\t")
    
                    elif flag['contract']:
    
                        path = '%s/%s/1_subduced-data/' % (outpath, ensemble)
    
                        filename = '/%s_p%1i_%s.h5' % (diagram, p_cm_sq, irrep)
                        subduced_data[diagram] = utils.read_hdf5_correlators(
                          path + filename, 'data')
    
                ######################################################################
                # Wick contraction
    
                if flag['contract']:
                    print '\tContracting data for %s' % correlator
    
                    if process == 'rho':
                        # rho analysis
                        contracted_data = wick.rho(
                          subduced_data, correlator, verbose)
                    elif process == 'pipi':
                        # pipi I=2 analysis
                        contracted_data = wick.pipi(
                          subduced_data, correlator, verbose)
                    del subduced_data
    
                    # write data to disc
                    path = '%s/%s/2_contracted-data/' % (outpath, ensemble)
                    filename = '/%s_p%1i_%s.h5' % (correlator, p_cm_sq, irrep)
                    utils.write_hdf5_correlators(
                      path, filename, contracted_data, 'data', verbose)
    
                elif (flag['create gevp'] or flag['plot']):
    
                    path = '%s/%s/2_contracted-data/' % (outpath, ensemble)
    
                    filename = '/%s_p%1i_%s.h5' % (correlator, p_cm_sq, irrep)
                    contracted_data = \
                      utils.read_hdf5_correlators(path+filename, 'data')
    
                ######################################################################
                # Plotting correlators seperately
    
                if flag['contract'] and flag['plot']:
    
                    if verbose:
                        print "Plotting data for ", irrep, correlator
    
                    path = '%s/%s/4_plots/p%1i/%s/' % (outpath, ensemble, p_cm_sq, irrep)
    
                    gevp_labels = ['gevp_row', 'gevp_col']
                    momentum_labels = \
                      [col for col in contracted_data.index.names if 'p^{' in col]
                    gamma_labels = \
                      [col for col in contracted_data.index.names if 'gamma' in col]
    
                    if plot_experimental:
    
                        # sum over equivalent momenta
                        plotdata_tmp = contracted_data.apply(np.real)
    
                        filename = './%s_p%1i_%s_gammas_%s.html' % (
                          correlator, p_cm_sq, irrep, continuum_basis_string)
                        bokeh.experimental(plotdata_tmp, correlator, bootstrapsize,
                          filename, logscale, verbose)
    
                    if plot_avg:
                        plotdata_tmp = contracted_data.apply(np.real)
                        # sum over momenta and gamma structures
                        plotdata_tmp = plotdata_tmp.sum(
                          level=gevp_labels + ['p_{cm}', '\mu'])
                        plotdata_tmp = plotdata_tmp.mean(level=gevp_labels)
    
                        filename = '/%s_p%1i_%s_avg_%s.pdf' % (
                          correlator, p_cm_sq, irrep, continuum_basis_string)
                        pdfplot = utils.create_pdfplot(path, filename)
                        plot.for_each_gevp_element(plot.average, plotdata_tmp,
                            bootstrapsize, pdfplot, logscale, verbose)
                        pdfplot.close()
    
    
                    if plot_pcm_and_mu:
                        plotdata_tmp = contracted_data.apply(np.real)
                        # sum over momenta and gamma structures
                        plotdata_tmp = plotdata_tmp.sum(
                          level=gevp_labels + ['p_{cm}', '\mu'])
    
                        filename = '/%s_p%1i_%s_pcm-and-mu_%s.pdf' % (
                          correlator, p_cm_sq, irrep, continuum_basis_string)
                        pdfplot = utils.create_pdfplot(path, filename)
                        plot.for_each_gevp_element(plot.pcm_and_mu, plotdata_tmp,
                            bootstrapsize, pdfplot, logscale, verbose)
                        pdfplot.close()
    
                    if plot_p_and_g:
                        # Want to look at all individual contributions
                        # Todo: Want to have imaginary part as well
                        plotdata_tmp = contracted_data.copy()
                        filename = '/%s_p%1i_%s_gammas_%s.pdf' % (
                          correlator, p_cm_sq, irrep, continuum_basis_string)
                        pdfplot = utils.create_pdfplot(path, filename)
                        plot.p_and_gammas(plotdata_tmp, correlator, bootstrapsize,
                              pdfplot, logscale, verbose)
                        pdfplot.close()
    
                if flag['create gevp']:
    
                    # Only real part is physically relevant at that point
                    contracted_data_avg[correlator] = contracted_data.apply(
                      np.real)
    
                    gevp_labels = ['gevp_row', 'gevp_col']
                    momentum_labels = \
                      [col for col in contracted_data.columns if 'p^{' in col]
                    gamma_labels = \
                      [col for col in contracted_data.columns if 'gamma' in col]
    
                    # sum over gamma structures.
                    contracted_data_avg[correlator] = contracted_data_avg[correlator].sum(
                      level=gevp_labels + ['p_{cm}'] + ['\mu'] + momentum_labels)
                    # sum over equivalent momenta
                    contracted_data_avg[correlator] = contracted_data_avg[correlator].sum(
                      level=gevp_labels + ['p_{cm}'] + ['\mu'])
    
            ##########################################################################
            # Gevp construction
    
            if flag['create gevp']:
    
                print '\tCreating gevp'
    
                gevp_data = setup_gevp.build_gevp(contracted_data_avg, process,
                                  verbose)
    
                path = '%s/%s/3_gevp-data/' % (outpath, ensemble)
                filename = '%s_p%1i_%s_%d.h5' % (process, p_cm_sq, irrep, 1)
                utils.write_hdf5_correlators(path, filename, gevp_data, 'data', verbose)
    
                # average over rows and  p_cm_sq.
                gevp_data_avg = gevp_data.mean(level=gevp_labels)
    
                path = '%s/%s/3_gevp-data/p%1i/%s/' % (outpath, ensemble, p_cm_sq, irrep)
                utils.write_ascii_gevp(path, process, gevp_data_avg, verbose)
    
            ##########################################################################
            # Plotting everything together
    
            if flag['create gevp'] and flag['plot']:
    
                if verbose:
                    print "Plotting Gevp data for ", irrep
    
                path = '%s/%s/4_plots/p%1i/%s/' % (outpath, ensemble, p_cm_sq, irrep)
    
                if plot_avg:
                    filename = 'Gevp_p%1d_%s_avg.pdf' % (p_cm_sq, irrep)
                    pdfplot = utils.create_pdfplot(path, filename)
                    plot.gevp(plot.average, gevp_data_avg, bootstrapsize, pdfplot,
                        logscale, verbose)
                    pdfplot.close()
    
                if plot_pcm_and_mu:
                    filename = 'Gevp_p%1d_%s_pcm-and-mu.pdf' % (p_cm_sq, irrep)
                    pdfplot = utils.create_pdfplot(path, filename)
                    plot.gevp(plot.pcm_and_mu, gevp_data, bootstrapsize, pdfplot,
                        logscale, verbose)
                    pdfplot.close()
