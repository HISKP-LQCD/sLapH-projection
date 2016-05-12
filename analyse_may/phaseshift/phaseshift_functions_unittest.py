"""
Unit tests for the phaseshift functions.
"""

import unittest
import numpy as np

from phaseshift_functions import calculate_phaseshift

class Phase_Test(unittest.TestCase):
    def test_cmf_T1(self):
        q = 0.1207*24/(2.*np.pi)
        delta, tandelta, sindelta = calculate_phaseshift(q*q, irrep="T1")
        self.assertAlmostEqual(delta*180./np.pi, 136.65, delta=0.01)

    def test_mf1_A1(self):
        d2 = 1
        L = 32
        q = 0.161*L/(2.*np.pi)
        E = 0.440
        Ecm = 0.396
        gamma = E/Ecm
        delta, tandelta, sindelta = calculate_phaseshift(q*q, gamma, d2)
        self.assertAlmostEqual(delta*180./np.pi, 118.75, delta=0.01)
        #self.assertAlmostEqual(delta*180./np.pi, 115.74, delta=0.01)

    def test_mf2(self):
        d2 = 2
        L = 32
        q = 0.167*L/(2.*np.pi)
        E = 0.490
        Ecm = 0.407
        gamma = E/Ecm
        delta, tandelta, sindelta = calculate_phaseshift(q*q, gamma, d2)
        self.assertAlmostEqual(delta*180./np.pi, 128.57, delta=0.01)
        #self.assertAlmostEqual(delta*180./np.pi, 127.99, delta=0.01)

    def test_cmf_multi(self):
        q = np.ones((100,)) * 0.1207*24/(2.*np.pi)
        delta, tandelta, sindelta = calculate_phaseshift(q*q, irrep="T1")
        self.assertTrue(np.allclose(delta*180./np.pi, 136.65, atol=0.01))

if __name__ == "__main__":
    unittest.main()

