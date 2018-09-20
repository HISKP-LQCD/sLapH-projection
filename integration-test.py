import argparse
import glob
import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import os
import sys
import tempfile
import tarfile
from urllib2 import urlopen, URLError, HTTPError
import shutil
import numpy as np

import src.utils as utils
import src.infile as infile
from src.main import *

class TestIntegration(unittest.TestCase):

    def setUp(self):

        self.outpath = tempfile.mkdtemp()

        if args.datapath == None:
           self.datapath = tempfile.mkdtemp()
        else:
            self.datapath = args.datapath
            if not os.path.isdir(self.datapath):
                print 'datapath does not exist'
                abort()

        self.ensemble = 'integration'

        # Download raw data from url
        if not args.no_download:
            url = "https://www.itkp.uni-bonn.de/~werner/sLapH-projection_integration-test_data/A40.24-cnfg0714.tar" 
    
            # Taken from stackoverflow and modified
            # https://stackoverflow.com/questions/4028697/how-do-i-download-a-zip-file-in-python-using-urllib2
            try:
                f = urlopen(url)
                print "downloading " + url
        
                with open(self.datapath +  '/' + os.path.basename(url), "wb") as local_file:
                    local_file.write(f.read())
    
            except HTTPError, e:
                print "HTTP Error:", e.code, url
            except URLError, e:
                print "URL Error:", e.reason, url
    
            tar = tarfile.open(self.datapath +  '/' + os.path.basename(url))
            tar.extractall(self.datapath)
            tar.close()

        try:
            os.makedirs(self.outpath + '/' + self.ensemble)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        for filename in glob.glob(r'tests/integration/*.ini'):
            shutil.copy(filename, 
                self.outpath + '/' + self.ensemble + '/')

    def tearDown(self):

        shutil.rmtree(self.outpath)
        if args.datapath == None:
            shutil.rmtree(self.datapath)

    def testPi(self):

        test_parameters = infile.read('tests/integration/pi.ini', None, None, verbose=0)
        # Modify test parameters to work with temporary paths
        test_parameters.update({'outpath' : self.outpath})
        test_parameters.update({'ensemble' : self.ensemble})
        test_parameters.update(
            {'directories' : [self.datapath+'/A40.24/'] * len(test_parameters['list_of_diagrams'])})

        main(continuum_basis_string='marcus-con', verbose=0, **test_parameters)

        calculated = utils.read_hdf5_correlators(test_parameters['outpath'] + '/' + 
                test_parameters['ensemble'] + '/3_gevp-data/pi_p0_A1g.h5')

        expected = utils.read_hdf5_correlators('tests/integration/pi_p0_A1g.h5')

        print 'Compare ', test_parameters['outpath'] + '/' + \
                test_parameters['ensemble'] + '/3_gevp-data/pi_p0_A1g.h5', ' with ', \
                'tests/integration/pi_p0_A1g.h5'
        
        assert_frame_equal(expected, calculated)


    def testRho(self):

        test_parameters = infile.read('tests/integration/rho.ini', None, None, verbose=0)
        # Modify test parameters to work with temporary paths
        test_parameters.update({'outpath' : self.outpath})
        test_parameters.update({'ensemble' : self.ensemble})
        test_parameters.update(
            {'directories' : [self.datapath+'/A40.24/'] * len(test_parameters['list_of_diagrams'])})

        main(continuum_basis_string='marcus-con', verbose=0, **test_parameters)

        calculated = utils.read_hdf5_correlators(test_parameters['outpath'] + '/' + 
                test_parameters['ensemble'] + '/3_gevp-data/rho_p1_A1.h5')

        expected = utils.read_hdf5_correlators('tests/integration/rho_p1_A1.h5')

        print 'Compare ', test_parameters['outpath'] + '/' + \
                test_parameters['ensemble'] + '/3_gevp-data/rho_p1_A1.h5', ' with ', \
                'tests/integration/rho_p1_A1.h5'

        assert_frame_equal(expected, calculated)

# Taken from 
# https://stackoverflow.com/questions/44236745/parse-commandline-args-in-unittest-python
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_download", help="Flag whether data is downloaded", 
                        action='store_true')
    parser.add_argument("--datapath", default=None, help="path to test ensemble")
    
    ns, args = parser.parse_known_args(namespace=unittest)
    #args = parser.parse_args()
    return ns, sys.argv[:1] + args


if __name__ == '__main__':

    args, argv = parse_args()   # run this first
    print(args, argv)
    sys.argv[:] = argv       # create cleans argv for main()
    unittest.main()    
