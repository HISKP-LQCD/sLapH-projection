from ast import literal_eval
import ConfigParser
import numpy as np
import collections
import os
import pandas as pd
from pandas import Series, DataFrame
import sys


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

    # Angular momentum for particles of interest
    if process == 'rho':
        j = 1
    elif process == 'pi':
        j = 0

    path = os.path.join(outpath, ensemble)
    for p_cm_sq in list_of_pcm_sq:
        if verbose:
            print 80 * '#'
            print 'p_cm_sq = ', p_cm_sq

        ########################################################################
        # Read diagrams for contributing correlators
        if flag['read']:
            raw_data.read(
                process,
                path,
                T,
                list_of_diagrams,
                directories,
                sta_cnfg,
                end_cnfg,
                del_cnfg,
                missing_configs,
                p_cm_sq,
                p_cutoff,
                gamma_input,
                verbose)

        ########################################################################
        # Setup of list of pcm and irrep to loop over
        filename = os.path.join(
            path, '0_raw-data', '%s_p%1i_%s_qn.h5' %
            (process, p_cm_sq, list_of_diagrams[0]))
        lookup_qn = utils.read_hdf5_correlators(filename)

        list_of_irreps = projection.get_list_of_irreps(
            lookup_qn['p_{cm}'].unique(), path_to_sc, j)
        correlators = wick.set_lookup_correlators(list_of_diagrams)

        for irrep in list_of_irreps:
            print '\t  Subducing into %s' % irrep

            ##########################################################################
            # Handling of gevp parameters
            gevp_parameters = ConfigParser.SafeConfigParser(
                {'list of p_cm': default_list_of_pcm,
                 'list of q': default_list_of_q,
                 'beta': default_beta})

            infilename = os.path.join(path, '%s_p%1d_%s.ini' % (process, p_cm_sq, irrep))
            if(gevp_parameters.read(infilename) == []):
                print 'Did not find gevp parameters for {} in '\
                    '{}. Continuing with default values'.\
                    format(irrep, infilename)
                list_of_pcm = default_list_of_pcm
                list_of_q = default_list_of_q
                beta = default_beta
            else:
                list_of_pcm = gevp_parameters.get('gevp parameters', 'list of p_cm')
                list_of_q = gevp_parameters.get('gevp parameters', 'list of q')
                beta = gevp_parameters.get('gevp parameters', 'beta')

            # helper function to read all subduced data from disk
            if list_of_pcm is None:
                list_of_pcm = lookup_qn['p_{cm}'].apply(literal_eval).unique()
            else:
                list_of_pcm = [literal_eval(p_cm.strip())
                               for p_cm in list_of_pcm.split(';')]
            assert all([utils._abs2(p) == p_cm_sq for p in list_of_pcm]), \
                'list_of_pcm contains momenta with the wrong absolute value squared'
            list_of_pcm = [str(p) for p in list_of_pcm]

            if list_of_q is not None:
                list_of_q = [literal_eval(q.strip()) for q in list_of_q.split(';')]

            contracted_data_avg = {}
            for correlator, diagrams in correlators.iteritems():

                ######################################################################
                # Projection

                lattice_operators = projection.project(
                    j,
                    correlator,
                    continuum_basis_string,
                    gamma_input,
                    list_of_pcm,
                    list_of_q,
                    path_to_sc,
                    path_to_sc_2,
                    verbose)
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

                        filename = os.path.join(
                            path, '0_raw-data', '%s_p%1i_%s.h5' %
                            (process, p_cm_sq, diagram))
                        data = utils.read_hdf5_correlators(filename)
                        filename = os.path.join(
                            path, '0_raw-data', '%s_p%1i_%s_qn.h5' %
                            (process, p_cm_sq, diagram))
                        lookup_qn = utils.read_hdf5_correlators(filename)

                        lookup_corr = subduction.set_lookup_corr(
                            lattice_operators, lookup_qn, verbose)

                        subduced_data[diagram] = subduction.project_correlators(
                            data, lookup_corr)
                        del data

                        # Write data and lattice operators to disc
                        filename = os.path.join(
                            path, '1_subduced-data', '%s_p%1i_%s_%s.h5' %
                            (process, p_cm_sq, irrep, diagram))
                        utils.write_hdf5_correlators(
                            filename, subduced_data[diagram], verbose)

                        filename = os.path.join(
                            path,
                            '1_subduced-data',
                            '{}_{}_p{}_{}_operators.tsv'.format(
                                process,
                                diagram,
                                p_cm_sq,
                                irrep))
                        lattice_operators.sort_index().\
                            reset_index(['operator_label_{so}', 'operator_label_{si}'],
                                        drop=True).\
                            to_csv(filename, sep="\t")

                    elif flag['contract']:

                        filename = os.path.join(
                            path, '1_subduced-data', '%s_p%1i_%s_%s.h5' %
                            (process, p_cm_sq, irrep, diagram))
                        subduced_data[diagram] = utils.read_hdf5_correlators(filename)

                ######################################################################
                # Wick contraction

                if flag['contract']:
                    print '\tContracting data for %s' % correlator

                    contracted_data = wick.contract_correlators(
                        process, subduced_data, correlator, verbose)
                    del subduced_data

                    # write data to disc
                    filename = os.path.join(
                        path, '2_contracted-data', '%s_p%1i_%s_%s.h5' %
                        (process, p_cm_sq, irrep, correlator))
                    utils.write_hdf5_correlators(filename, contracted_data, verbose)

                elif (flag['create gevp'] or flag['plot']):

                    filename = os.path.join(
                        path, '2_contracted-data', '%s_p%1i_%s_%s.h5' %
                        (process, p_cm_sq, irrep, correlator))
                    contracted_data = utils.read_hdf5_correlators(filename)

                ######################################################################
                # Plotting correlators seperately

                if flag['contract'] and flag['plot']:

                    if verbose:
                        print "Plotting data for ", irrep, correlator

                    gevp_labels = ['gevp_row', 'gevp_col']
                    momentum_labels = \
                        [col for col in contracted_data.index.names if 'p^{' in col]
                    gamma_labels = \
                        [col for col in contracted_data.index.names if 'gamma' in col]

                    if plot_experimental:

                        # sum over equivalent momenta
                        plotdata_tmp = contracted_data.apply(np.real)

                        filename = './%s_p%1i_%s_%s_gammas.html' % (
                            process, p_cm_sq, irrep, correlator)
                        bokeh.experimental(plotdata_tmp, correlator, bootstrapsize,
                                           filename, logscale, verbose)

                    if plot_avg:
                        plotdata_tmp = contracted_data.apply(np.real)
                        # sum over momenta and gamma structures
                        plotdata_tmp = plotdata_tmp.sum(
                            level=gevp_labels + ['p_{cm}', '\mu'])
                        plotdata_tmp = plotdata_tmp.mean(level=gevp_labels)

                        filename = os.path.join(path, '4_plots', '%s_p%1i_%s_%s_avg.pdf' % (
                            process, p_cm_sq, irrep, correlator))
                        pdfplot = utils.create_pdfplot(filename)
                        plot.for_each_gevp_element(
                            plot.average, plotdata_tmp, bootstrapsize, pdfplot, logscale, verbose)
                        pdfplot.close()

                    if plot_pcm_and_mu:
                        plotdata_tmp = contracted_data.apply(np.real)
                        # sum over momenta and gamma structures
                        plotdata_tmp = plotdata_tmp.sum(
                            level=gevp_labels + ['p_{cm}', '\mu'])

                        filename = os.path.join(path, '4_plots', '%s_p%1i_%s_%s_pcm-and-mu.pdf' % (
                            process, p_cm_sq, irrep, correlator))
                        pdfplot = utils.create_pdfplot(filename)
                        plot.for_each_gevp_element(
                            plot.pcm_and_mu, plotdata_tmp, bootstrapsize, pdfplot, logscale, verbose)
                        pdfplot.close()

                    if plot_p_and_g:
                        # Want to look at all individual contributions
                        # Todo: Want to have imaginary part as well
                        plotdata_tmp = contracted_data.copy()
                        filename = os.path.join(path, '4_plots', '%s_p%1i_%s_%s_gammas.pdf' % (
                            process, p_cm_sq, irrep, correlator))
                        pdfplot = utils.create_pdfplot(filename)
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

                filename = os.path.join(
                    path, '3_gevp-data', '%s_p%1i_%s.h5' % (process, p_cm_sq, irrep))
                utils.write_hdf5_correlators(filename, gevp_data, verbose)

                basename = '%s_p%1i_%s' % (process, p_cm_sq, irrep)
                utils.write_ascii_gevp(
                    os.path.join(path, '3_gevp-data'), basename, gevp_data, verbose)

                # average over rows and  p_cm_sq.
                gevp_data_avg = gevp_data.mean(level=gevp_labels)

            ##########################################################################
            # Plotting everything together

            if flag['create gevp'] and flag['plot']:

                if verbose:
                    print "Plotting Gevp data for ", irrep

                if plot_avg:
                    filename = os.path.join(
                        path, '4_plots', '%s_p%1d_%s_gevp_avg.pdf' %
                        (process, p_cm_sq, irrep))
                    pdfplot = utils.create_pdfplot(filename)
                    plot.gevp(plot.average, gevp_data_avg, bootstrapsize, pdfplot,
                              logscale, verbose)
                    pdfplot.close()

                if plot_pcm_and_mu:
                    filename = os.path.join(path, '4_plots', '%s_p%1d_%s_gevp_pcm-and-mu.pdf' % (
                        process, p_cm_sq, irrep))
                    pdfplot = utils.create_pdfplot(filename)
                    plot.gevp(plot.pcm_and_mu, gevp_data, bootstrapsize, pdfplot,
                              logscale, verbose)
                    pdfplot.close()
