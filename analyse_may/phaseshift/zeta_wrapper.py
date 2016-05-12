"""
wrapper for the zeta function.
"""

import numpy as np
import memoize
import zeta

def Z(q2, gamma=None, l=0, m=0, d=np.array([0., 0., 0.]), m_split=1.,
        prec=10e-6, verbose=0):
    """Calculates the Luescher Zeta function.

    Parameters
    ----------
    q2 : float or ndarray
        The squared momentum transfer.
    gamma : float or ndarray, optional
        The Lorentz boost factor.
    l, m : ints, optional
        The orbital and magnetic quantum numbers.
    d : ndarray, optional
        The total momentum vector of the system.
    m_split : float, optional
        The mass difference between the particles.
    prec : float, optional
        The calculation precision.
    verbose : int
        The amount of info printed.
    """
    if isinstance(q2, (tuple, list, np.ndarray)):
        _q2 = np.asarray(q2)
        if gamma == None:
            _gamma = np.ones_like(_q2)
        else:
            _gamma = np.asarray(gamma)
        # use numpy iterator for to access all elements
        res = np.zeros_like(_q2, dtype=complex)
        it = np.nditer([_q2, _gamma, res], op_flags = [["readonly"],
            ["readonly"], ["writeonly", "no_broadcast"]])
        for q, g, r in it:
            r[...] = zeta.Z(q, g, l, m, d, m_split, prec, verbose)

        return it.operands[2]
    else:
        if gamma == None:
            return zeta.Z(q2, 1., l, m, d, m_split, prec, verbose)
        else:
            return zeta.Z(q2, gamma, l, m, d, m_split, prec, verbose)

def omega(q2, gamma=None, l=0, m=0, d=np.array([0., 0., 0.]), m_split=1.,
        prec=10e-6, exFac=False, verbose=0):
    """Calculates the Zeta function including the some prefactor.

    Parameters
    ----------
    q2 : float or ndarray
        The squared momentum transfer.
    gamma : float or ndarray, optional
        The Lorentz boost factor.
    l, m : ints, optional
        The orbital and magnetic quantum numbers.
    d : ndarray, optional
        The total momentum vector of the system.
    m_split : float, optional
        The mass difference between the particles.
    prec : float, optional
        The calculation precision.
    exFac : bool, optional
        excludes 1/(gamma*q*sqrt(PI)^3) from the result
    verbose : int
        The amount of info printed.

    Returns
    -------
    float or ndarray
        The value of the Zeta function.
    """
    if exFac:
        if l is not 0:
            factor = np.power(np.sqrt(q2), l) * np.sqrt(2*l)
        else:
            factor = 1.
    else:
        factor = gamma*np.power(np.sqrt(q2), l+1)*np.power(np.pi, 1.5)*\
                 np.sqrt(2*l+1)
    var =  Z(q2, gamma, l, m, d, m_split, prec, verbose)
    return var / factor

if __name__ == "main":
    pass
