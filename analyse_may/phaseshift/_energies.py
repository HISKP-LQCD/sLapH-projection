"""
Funtions for energy and momentum calculations.
"""

import numpy as np

from memoize import memoize

@memoize(50)
def EfromW(W, d2=0, L=24):
    """Calculates the moving frame energy from the CM energy.

    Parameters
    ----------
    W : float or ndarray
        The CM energy.
    d2 : int, optional
        The squared total three momentum of the system.
    L : int, optional
      The lattice size.

    Returns
    -------
    float or ndarray
        The energy.
    """
    return np.sqrt(W*W - d2 * 4. * np.pi*np.pi / (float(L)* float(L)))

def calc_Ecm(E, d2=0, L=24):
    """Calculates the center of mass energy and the boost factor.

    Calculates the Lorentz boost factor and the center of mass energy
    for moving frames.
    For the lattice dispersion relation see arXiv:1011.5288.

    Parameters
    ----------
    E : float or ndarray
        The energy of the system.
    d2 : int, optional
        The squared total three momentum of the system.
    L : int, optional
        The lattice size.

    Returns:
    float or ndarray
        The boost factor.
    float or ndarray
        The center of mass energy.
    """
    # if the data is from the cm system, return immediately
    if (d2 == 0):
        gamma = np.ones_like(E)
        return gamma, E
    Ecm = EfromW(E, d2, L)
    gamma = E / Ecm
    return gamma, Ecm

def q2fromE_mass(E, m, L=24):
    """Caclulates the q2 from the energy and the particle mass.

    Parameters
    ----------
    E : float or ndarray
        The energy of the particle.
    m : float or ndarray
        The particle mass.
    L : int, optional
        The lattice size.

    Returns
    -------
    float or ndarray
        The CM momentum squared.
    """
    return (0.25*E*E - m*m) * (float(L) / (2. * np.pi))**2

def calc_q2(E, m, L=24):
    """Calculates the momentum squared.

    Calculates the difference in momentum between interaction and non-
    interacting systems. The energy must be the center of mass energy.

    Parameters
    ----------
    E : float or ndarray
        The energy of the particle.
    m : float or ndarray
        The particle mass.
    L : int, optional
        The lattice size.

    Returns
    -------
    float or ndarray
        The CM momentum squared.
    """
    q2 = q2fromE_mass(E, m, L)

    return q2
