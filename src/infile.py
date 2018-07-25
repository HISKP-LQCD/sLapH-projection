import ConfigParser

##################################################################################
# Reading infile #################################################################

def read(infile, verbose):

    config = ConfigParser.SafeConfigParser(
            {'list of p_cm' : None,
                'Dirac structure' : 'gamma_i',
                'list of q' : None,
                'beta' : '1'})
    
    if(config.read(infile) == []):
        print "Error! Could not open infile: ", infile
        exit(-1)
    
    # optional parameter, Default: False
    flag_pion = config.getboolean('parameters', 'Read pion')
    
    # Choices: ['rho', 'pipi']
    process = config.get('parameters', 'Process to analyse').lower()
    if process not in ['rho', 'pipi']:
        print 'Only analysis for "rho" and "pipi" are supported!'
        sys.exit(0)
    
    if verbose:
        print 80 * '#'
        print 'Reading infile'
    
    if verbose >= 2:
        print 'Reading data for pion: {}'.format(flag_pion)
        print 'Process to analyse: {}'.format(process)
    
    flag = {}
    flag['read'] = config.getboolean('parameters', 'Read rho')
    flag['subduce'] = config.getboolean('parameters', 'Subduce')
    flag['contract'] = config.getboolean('parameters', 'Contract')
    flag['create gevp'] = config.getboolean('parameters', 'Create Gevp')
    flag['plot'] = config.getboolean('parameters', 'Plot data')
    
    if verbose >= 2:
        print flag
    
    sta_cnfg = config.getint('gauge configuration numbers',
                 'First configuration')
    end_cnfg = config.getint('gauge configuration numbers',
                 'Last configuration')
    del_cnfg = config.getint('gauge configuration numbers',
                 'Configuration stepping')
    missing_configs = config.get('gauge configuration numbers',
                   'Missing configurations')
    # turns missing configs into list of integers
    if (missing_configs == ''):
        missing_configs = []
    else:
        missing_configs = [int(m) for m in missing_configs.split(',')]
    
    if verbose >= 2:
        print 'Reading configs', sta_cnfg, '-', end_cnfg, ' in steps of ', del_cnfg
        print 'Skipping ', missing_configs
    
    ensemble = config.get('ensemble and frame', 'Ensemble Name')
    T = config.getint('ensemble and frame', 'T')
    list_of_pcm_sq = config.get('ensemble and frame', 'p_cm^2')
    list_of_pcm_sq = [int(p) for p in list_of_pcm_sq.split(',')]
    p_cutoff = config.getint('ensemble and frame', 'p_cutoff')
    
    if verbose >= 2:
        print ensemble
        print T
        print 'p_cm = ', p
        print 'p_cutoff = ', p_cutoff
    
    default_list_of_pcm = config.get('gevp parameters', 'list of p_cm')
    gamma_input = config.get('gevp parameters', 'Dirac structure')
    # translates list of names for gamma structures to indices used in
    # contraction code
    gamma_input = gamma_input.replace(" ", "").split(';')
    gamma_input = {0: ['gamma_5'], 1: gamma_input}
    default_list_of_q = config.get('gevp parameters', 'list of q')
    default_beta = config.get('gevp parameters', 'beta')
    
    if verbose >= 2:
        print 'list of pcm = ', default_list_of_pcm
        print 'Gamma structures: ', gamma_input
        print 'list of q = ', default_list_of_q
        print 'beta = ', default_beta
    
    list_of_diagrams = config.get('contraction details', 'Diagram')
    list_of_diagrams = list_of_diagrams.replace(" ", "").split(',')
    directories = config.get('contraction details', 'Input Path')
    directories = directories.replace(" ", "").replace("\n", "")
    directories = directories.split(',')
    
    if verbose >= 2:
        print 'Wick diagrams to subduce: ', list_of_diagrams
    # use the same directory for all list_of_diagrams if only one is given
    if (len(directories) == 1):
        directories = directories * len(list_of_diagrams)
    if verbose >= 2:
        for i in range(len(list_of_diagrams)):
            print 'Read data for ', list_of_diagrams[i], ' from ', directories[i]
    
    path_to_sc = config.get('Environment details',
                'Path to one-meson subduction coefficients')
    path_to_sc_2 = config.get('Environment details',
                'Path to two-meson subduction coefficients')
    
    outpath = config.get('Environment details', 'Output Path')
    
    if verbose >= 2:
        print 'Subduction coefficients will be read from ', path_to_sc
        print 'Two-particle projection coefficients will be read from ', path_to_sc_2
        print 'Data will be writen to', outpath
    
    plot_p_and_g = config.getboolean('plot details', 'Plot correlators')
    plot_pcm_and_mu = config.getboolean('plot details', 'Plot p_cm and row')
    plot_avg = config.getboolean('plot details', 'Plot average')
    
    plot_experimental = config.getboolean('plot details', 'Plot experimental')
    
    logscale = config.getboolean('plot details', 'Logscale')
    
    bootstrapsize = config.getint('plot details', 'Number of bootstrap samples')
    
    if verbose >= 2:
        if plot_p_and_g:
            print 'Plotting correlators for all momenta and Dirac structures'
        if plot_pcm_and_mu:
            print 'Plotting correlators for all p_cm_and_mu'
        print 'Logscale: ', logscale
    
    return flag_pion, process, flag, sta_cnfg, end_cnfg, del_cnfg, missing_configs, ensemble, \
        T, list_of_pcm_sq, p_cutoff, default_list_of_pcm, gamma_input, default_list_of_q, \
        default_beta, list_of_diagrams, directories, path_to_sc, path_to_sc_2, outpath, \
        plot_p_and_g, plot_pcm_and_mu, plot_avg, plot_experimental, logscale, \
        bootstrapsize
   
