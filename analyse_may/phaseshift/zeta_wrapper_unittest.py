"""
Unit tests for the zeta function wrappers.
"""

import unittest
import numpy as np

from zeta_wrapper import Z, omega

class Zeta_Test(unittest.TestCase):
    def test_cmf(self):
        Pcm = np.array([0., 0., 0.])
        q = 0.1207*24/(2.*np.pi)
        gamma = 1.0
        zeta = Z(q*q, gamma, d = Pcm).real
        delta = np.arctan(np.pi**(3./2.)*q/zeta)*180./np.pi
        if delta < 0:
            delta = 180+delta
        self.assertAlmostEqual(delta, 136.65, delta=0.01)

    def test_mf1(self):
        Pcm = np.array([0., 0., 1.])
        L = 32
        q = 0.161*L/(2.*np.pi)
        E = 0.440
        Ecm = 0.396
        gamma = E/Ecm
        Z00 = Z(q*q, gamma, d = Pcm).real
        Z20 = Z(q*q, gamma, d = Pcm, l = 2).real
        delta = np.arctan(gamma*np.pi**(3./2.) * q / \
                (Z00 + (2./(q*q*np.sqrt(5)))*Z20))*180./np.pi
        if delta < 0:
            delta = 180+delta
        self.assertAlmostEqual(delta, 115.74, delta=0.01)

    def test_mf2(self):
        Pcm = np.array([1., 1., 0.])
        L = 32
        q = 0.167*L/(2.*np.pi)
        E = 0.490
        Ecm = 0.407
        gamma = E/Ecm
        Z00 = Z(q*q, gamma, d = Pcm).real
        Z20 = Z(q*q, gamma, d = Pcm, l = 2).real
        Z22  = Z(q*q, gamma, d = Pcm, l = 2, m = 2).imag
        Z2_2 = Z(q*q, gamma, d = Pcm, l = 2, m = -2).imag
        delta = np.arctan(gamma*np.pi**(3./2.) * q / \
                (Z00 - (1./(q*q*np.sqrt(5)))*Z20 \
                + ((np.sqrt(3./10.)/(q*q))*(Z22-Z2_2))))*180./np.pi
        if delta < 0:
            delta = 180+delta
        self.assertAlmostEqual(delta, 127.99, delta=0.01)

    def test_cmf_multi(self):
        Pcm = np.array([0., 0., 0.])
        q = np.ones((100,)) * 0.1207*24/(2.*np.pi)
        gamma = np.ones((100,))
        zeta = Z(q*q, gamma, d = Pcm).real
        delta = np.arctan(np.pi**(3./2.)*q/zeta)*180./np.pi
        delta[delta < 0.] += 180.
        self.assertTrue(np.allclose(delta, 136.65, atol=0.01))

    def test_omega_cmf(self):
        Pcm = np.array([0., 0., 0.])
        q = 0.1207*24/(2.*np.pi)
        gamma = 1.0
        delta = np.arctan(1./omega(q*q, gamma, d=Pcm))*180./np.pi
        if delta < 0:
            delta = 180+delta
        self.assertAlmostEqual(delta, 136.65, delta=0.01)

if __name__ == "__main__":
    unittest.main()

