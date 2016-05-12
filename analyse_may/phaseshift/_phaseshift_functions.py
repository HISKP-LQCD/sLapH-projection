"""
Functions to calculate the phaseshift.
"""

import numpy as np
from zeta_wrapper import omega

def calc_ps(q2, gamma=None, d2=0, irrep="A1", prec=1e-5, debug=0):
    """Calculates the phase shift using Luescher's Zeta function.

    The names of the Irreps are the same as in M. Goeckeler et al., Phys. Rev.
    D 86, 094513 (2012).

    Parameters
    ----------
    q2 : float or ndarray
        The momentum shift squared.
    gamma : float or ndarray, optional
        The Lorentz factor for moving frames.
    d2 : int, optional
        The squared total three momentum of the system.
    irrep : str, optional
        The irrep for which to calculate, defaults to A1.
    prec : float, optional
        The precision of the Zeta function calculation.
    debug : int, optional
        The amount of information printed to screen.

    Returns:
        An array of the phase shift and tan(delta).
    """

    # changed call to depend on d2 rather irrep because it is easier to call
    if (d2 == 1) or (d2 == 2) or (d2 == 3) or (d2 == 4):
        return phaseshift_A1(q2, gamma, d2, prec, debug)
    elif (d2 == 0):
        return phaseshift_T1(q2, gamma, d2, prec, debug)
    else:
        raise RuntimeError("Moving frame %d not implemented" % d2)

#    if irrep == "A1":
#        return phaseshift_A1(q2, gamma, d2, prec, debug)
#    elif irrep == "T1":
#        return phaseshift_T1(q2, gamma, d2, prec, debug)
#    else:
#        raise RuntimeError("Irrep %s not implemented" % irrep)

def phaseshift_A1(q2, gamma=None, d2=0, prec=1e-5, debug=0):
    """Calculates the phase shift for irrep A1 using Luescher's Zeta function.

    The names of the Irreps are the same as in M. Goeckeler et al., Phys. Rev.
    D 86, 094513 (2012).

    Parameters
    ----------
    q2 : float or ndarray
        The momentum shift squared.
    gamma : float or ndarray, optional
        The Lorentz factor for moving frames.
    d2 : int, optional
        The squared total three momentum of the system.
    prec : float, optional
        The precision of the Zeta function calculation.
    debug : int, optional
        The amount of information printed to screen.

    Returns:
        An array of the phase shift and tan(delta).
    """
    if gamma == None:
        _gamma = np.ones_like(q2)
    else:
        _gamma = gamma
    # init calculation
    _pi3 = np.pi**3
    _num = _gamma * np.sqrt(_pi3 * q2)
    #CMF
    if d2 == 0:
        raise NotImplementedError("")
        #_den = omega(q2, _gamma, 0, 0, d, exFac=True).real
        #tandelta = _num / _den
        #delta = np.arctan( _num / _den) + np.pi
    # MF1
    elif d2 == 1 or d2 ==4:
        if d2 == 1:
            d = np.array([0., 0., 1.])
        else:
            d = np.array([0., 0., 2.])
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real + \
               2 * omega(q2, _gamma, 2, 0, d, exFac=True).real
        tandelta = _num / _den
        delta = np.arctan( _num / _den) + np.pi
    # MF2
    elif d2 == 2:
        d = np.array([1., 1., 0.])
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real -\
               omega(q2, _gamma, 2, 0, d, exFac=True).real +\
               np.sqrt(6) * omega(q2, _gamma, 2, 2, d, exFac=True).imag
        tandelta = _num / _den
        delta = np.arctan( _num / _den) + np.pi
    # MF3
    elif d2 == 3:
        d = np.array([1., 1., 1.])
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real +\
               2 * np.sqrt(6) * omega(q2, _gamma, 2, 2, d, exFac=True).imag
        tandelta = _num / _den
        delta = np.arctan( _num / _den) + np.pi
    else:
        raise RuntimeError("moving frame for d2 = %d not implemented" % d2)
    sindelta = np.sin(delta)**2
    return delta, tandelta, sindelta

def phaseshift_T1(q2, gamma=None, d2=0, prec=1e-5, debug=0):
    if gamma == None:
        _gamma = np.ones_like(q2)
    else:
        _gamma = gamma
    # init calculation
    _pi3 = np.pi**3
    _num = _gamma * np.sqrt(_pi3 * q2)
    #CMF
    if d2 == 0:
        d = np.array([0., 0., 0.])
        _den = omega(q2, _gamma, 0, 0, d, exFac=True).real
        tandelta = _num / _den
        delta = np.arctan( _num / _den) + np.pi
    else:
        raise RuntimeError("moving frame for d2 = %d not implemented" % d2)
    sindelta = np.sin(delta)**2
    return delta, tandelta, sindelta

if __name__ == "main":
    pass
