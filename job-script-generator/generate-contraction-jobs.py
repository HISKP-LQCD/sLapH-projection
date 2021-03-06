#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2017-2018 Martin Ueding <dev@martin-ueding.de>
# Licensed under the MIT/Expat license

import argparse
import errno
import os
import shutil
import sys

import jinja2
import numpy as np


# replaces os.makedirs(..., exist_ok=True) because python2
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def main():
    options = _parse_args()

    # Run the consistency checks before doing anything. After all checks are
    # through, we may do things that change files on disk.
    do_consistency_checks(options)

    # Create the main work directory.
    mkdir_p(options.rundir)
    mkdir_p(os.path.join(options.outdir, options.ensemble))

    # Write the command line parameters the script was exected with
    with open(os.path.join(options.rundir, 'generate-contraction-jobs.log'), 'w') as f:
        f.write("Job scripts where generated with command line options:\n\n")
        for arg in sys.argv:
            f.write("%s " % arg)
        f.write("\n")

    # Copy the start_runs script into the right spot.
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'start_runs.sh'), os.path.join(options.rundir, 'start_runs.sh'))

    # Load the templates from the directory that this script is located in.
    # This path can be queried from the zeroth command line argument.
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(
        os.path.dirname(os.path.abspath(sys.argv[0]))))

    # Create an ini-file to read the diagrams seperately
    template_rho = env.get_template('read.ini.j2')
    rho = 'rho.ini'
    rendered_rho = template_rho.render(
        process='Rho',
        conf_start=options.conf_start,
        conf_end=options.conf_end,
        conf_step=options.conf_step,
        conf_skip=options.conf_skip,
        ensemble=options.ensemble,
        T=options.T,
        datapath=options.datapath,
        codepath=os.path.dirname(options.exe),
        outpath=options.outdir
    )
    with open(os.path.join(options.rundir, rho), 'w') as f:
        f.write(rendered_rho)

    # Create a job script for the scheduler.
    template_jobscript = env.get_template('read_qbig_slurm.sh.j2')
    for momentum in range(5):
        for diagram in ['C20', 'C3c', 'C4cD', 'C4cB']:
            if diagram.startswith('C4'):
                memory = 252
            elif diagram.startswith('C3'):
                memory = 123
            else:
                memory = 56
            jobscriptfile = 'read_qbig_slurm_p{}_{}.sh'.format(momentum, diagram)
            rendered_jobscript = template_jobscript.render(
                rundir=options.rundir,
                executable=options.exe,
                jobname=options.jobname + '_' + options.ensemble + '_p' + str(momentum) + '_' + diagram,
                email_address=options.email,
                memory=memory,
                infile=os.path.join(options.rundir, rho),
                momentum=momentum,
                diagram=diagram
            )
            with open(os.path.join(options.rundir, jobscriptfile), 'w') as f:
                f.write(rendered_jobscript)

    ###################################################################################### 

    # Copy the files per ensemble into the right spot.
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p0_T1u.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p0_T1u.ini'))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p1_A1.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p1_A1.ini'))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p1_E.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p1_E.ini'))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p2_A1.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p2_A1.ini'))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p2_B1.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p2_B1.ini'))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p2_B2.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p2_B2.ini'))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p3_A1.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p3_A1.ini'))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p3_E.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p3_E.ini'))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p4_A1.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p4_A1.ini'))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'rho_p4_E.ini'), os.path.join(os.path.join(options.outdir, options.ensemble), 'rho_p4_E.ini'))


    # Create an ini-file for the subduction
    template_rho = env.get_template('general.ini.j2')
    rendered_rho = template_rho.render(
        process='Rho',
        conf_start=options.conf_start,
        conf_end=options.conf_end,
        conf_step=options.conf_step,
        conf_skip=options.conf_skip,
        ensemble=options.ensemble,
        T=options.T,
        datapath=options.datapath,
        codepath=os.path.dirname(options.exe),
        outpath=options.outdir
    )
    with open(os.path.join(os.path.join(options.outdir, options.ensemble), 'rho.ini'), 'w') as f:
        f.write(rendered_rho)

    # Create an ini-file for the pion gevp
    template_pi = env.get_template('general.ini.j2')
    rendered_pi = template_rho.render(
        process='Pi',
        conf_start=options.conf_start,
        conf_end=options.conf_end,
        conf_step=options.conf_step,
        conf_skip=options.conf_skip,
        ensemble=options.ensemble,
        T=options.T,
        datapath=options.datapath,
        codepath=os.path.dirname(options.exe),
        outpath=options.outdir
    )
    with open(os.path.join(os.path.join(options.outdir, options.ensemble), 'pi.ini'), 'w') as f:
        f.write(rendered_pi)

    # Create a job script for the scheduler.
    template_jobscript = env.get_template('project_qbig_slurm.sh.j2')
    jobscriptfile = 'project_qbig_slurm.sh'
    rendered_jobscript = template_jobscript.render(
        rundir=os.path.join(options.outdir, options.ensemble),
        executable=options.exe,
        jobname=options.jobname + '_' + options.ensemble,
        email_address=options.email,
    )
    with open(os.path.join(os.path.join(options.outdir, options.ensemble), jobscriptfile), 'w') as f:
        f.write(rendered_jobscript)


def do_consistency_checks(options):
    '''
    This runs various consistency checks.

    If something is off, an exception is raised with an explanation.
    '''
    # Skip the tests in case the user wants to ignore them.
    if options.ignore_checks:
        return

    # Check that the executable exists.
    if not os.path.isfile(options.exe):
        raise RuntimeError('The executable at “{}” does not exist! Please make sure that a correct path has been given.'.format(options.exe))

def _parse_args():
    parser = argparse.ArgumentParser(description='Generates a hierarchy of input files and job scripts for the contraction code. The script will also make sure that the referenced files actually exist such that the jobs will have a higher chance of succeeding.')

    parser.add_argument('--ignore-checks', action='store_true', help='Do not run the tests for existence of input files.')

    group_config = parser.add_argument_group('Configuration', 'Options that are inherent for the underlying gauge configurations.')
    group_config.add_argument('conf_start', type=int, help='First configuration, inclusive')
    group_config.add_argument('conf_end', type=int, help='Last configuration, inclusive')
    group_config.add_argument('conf_step', type=int, nargs='?', default=1, help='default: %(default)s')
    group_config.add_argument('--conf-skip', type=int, nargs='+', help='Skip the given gauge configurations.', default=[])

    group_ensemble = parser.add_argument_group('Ensemble', 'Options that discribe the physical choices made')
    group_ensemble.add_argument('--ensemble', required=True, help='Name of the gauge ensemble')
    group_ensemble.add_argument('--T', required=True, help='Time extent of the ensemble')
    group_ensemble.add_argument('--datapath', required=True, help='Path to contracted diagrams')
#    group_ensemble.add_argument('-p', '--momentum', type=int, nargs='+')
#    group_ensemble.add_argument('-d', '--diagram', nargs='+')

    group_job = parser.add_argument_group('Job', 'Options for the jobs to create.')
    group_job.add_argument('--rundir', required=True, help='Base path for infiles.')
    group_job.add_argument('--outdir', required=True, help='Base path for output.')
    group_job.add_argument('--exe', required=True, help='Path to the executable. This will be copied into the hierarchy to prevent accidential overwrites.')
    group_job.add_argument('--jobname', default='projection', help='Name of the submitted job. Default: %(default)s')
    group_job.add_argument('--email', default='', help='Email address to send job notifications to. If this is not given, no emails will be send.')
   
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    main()
