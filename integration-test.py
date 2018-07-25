
import glob
import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import os
import tempfile
import tarfile
from urllib2 import urlopen, URLError, HTTPError
import shutil

import src.utils as utils
import src.infile as infile
from src.main import *

class TestIntegration(unittest.TestCase):

    def setUp(self):

        self.outpath = tempfile.mkdtemp()
        self.datapath = tempfile.mkdtemp()
        self.ensemble = 'integration'

        # Download raw data from url
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
            os.makedirs(self.outpath + '/' + self.ensemble + '/3_gevp-data/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        for filename in glob.glob(r'tests/integration/*.ini'):
            shutil.copy(filename, 
                self.outpath + '/' + self.ensemble + '/3_gevp-data/')

    def tearDown(self):

        shutil.rmtree(self.outpath)
        shutil.rmtree(self.datapath)

    def testGevp(self):

        test_parameters = infile.read('tests/integration/A40.24.ini', verbose=0)
        # Modify test parameters to work with temporary paths
        test_parameters.update({'outpath' : self.outpath})
        test_parameters.update({'ensemble' : self.ensemble})
        test_parameters.update(
            {'directories' : [self.datapath+'/A40.24/'] * len(test_parameters['list_of_diagrams'])})

        main(continuum_basis_string='marcus-con', verbose=0, **test_parameters)

        calculated = utils.read_hdf5_correlators(test_parameters['outpath'] + '/' + 
                test_parameters['ensemble'] + '/3_gevp-data/rho_p1_A1_1.h5', 'data')

        expected = utils.read_hdf5_correlators('tests/integration/rho_p1_A1_1.h5', 'data')
        
        assert_frame_equal(expected, calculated)

if __name__ == '__main__':
    unittest.main()
