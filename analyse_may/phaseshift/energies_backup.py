"""
Funtions for energy and momentum calculations.
"""

import numpy as np

from memoize import memoize

@memoize(50)
def WfromE(E, d=np.array([0., 0., 0.]), L=24):
    """Calculates the CM energy from the energy.

    Parameters
    ----------
    E : float or ndarray
        The energy.
    d : ndarray, optional
        The total momentum vector of the system.
    L : int, optional
      The lattice size.

    Returns
    -------
    float or ndarray
        The center of mass energy.
    """
    return np.sqrt(E*E + np.dot(d, d) * 4. * np.pi*np.pi / (float(L)*float(L)))

@memoize(50)
def EfromW(W, d=np.array([0., 0., 0.]), L=24):
    """Calculates the moving frame energy from the CM energy.

    Parameters
    ----------
    W : float or ndarray
        The CM energy.
    d : ndarray, optional
        The total momentum vector of the system.
    L : int, optional
      The lattice size.

    Returns
    -------
    float or ndarray
        The energy.
    """
    return np.sqrt(W*W - np.dot(d, d) * 4. * np.pi*np.pi / (float(L)* float(L)))

@memoize(50)
def WfromE_lat(E, d=np.array([0., 0., 0.]), L=24):
    """Calculates the CM energy from the energy using the lattice dispersion
    relation.

    Parameters
    ----------
    E : float or ndarray
        The energy.
    d : ndarray, optional
        The total momentum vector of the system.
    L : int, optional
      The lattice size.

    Returns
    -------
    float or ndarray
        The center of mass energy.
    """
    return np.arccosh(np.cosh(E) + 2. * np.sum(np.sin(d*np.pi/float(L))**2))

@memoize(50)
def EfromW_lat(W, d=np.array([0., 0., 0.]), L=24):
    """Calculates the moving frame energy from the CM energy using the lattice
    dispersion relation.

    Parameters
    ----------
    W : float or ndarray
        The CM energy.
    d : ndarray, optional
        The total momentum vector of the system.
    L : int, optional
      The lattice size.

    Returns
    -------
    float or ndarray
        The energy.
    """
    return np.arccosh(np.cosh(W) - 2. * np.sum(np.sin(d*np.pi/float(L))**2))

@memoize(50)
def WfromMass(m, q, L=24):
    """Calculates the center of mass energy for a particle 
    with mass m and momentum q.

    Parameters
    ----------
    m : float or ndarray
        The particle mass
    q : float or ndarray
        The modulus of the momentum of the particle.
    L : int, optional
      The lattice size.

    Returns
    -------
    float or ndarray
        The particle energy.
    """
    return 2.*np.sqrt(m*m + 4.*q*q*np.pi*np.pi/(float(L)*float(L)))

@memoize(50)
def WfromMass_lat(m, q, L=24):
    """Calculates the center of mass energy for a particle 
    with mass m and momentum q using the lattice dispersion
    relation.

    Parameters
    ----------
    m : float or ndarray
        The particle mass
    q : float or ndarray
        The modulus of the momentum of the particle.
    L : int, optional
      The lattice size.

    Returns
    -------
    float or ndarray
        The particle energy.
    """
    return 2. * np.arccosh(np.cosh(m) + 2. * np.sin(q * np.pi / float(L))**2)

def calc_gamma(q2, m, d=np.array([0., 0., 1.]), L=24):
    """Calculates the Lorentz boost factor for the given energy and momentum.

    Parameters
    ----------
    q2 : float or ndarray
        The modulus of the momentum of the particle squared.
    m : float or ndarray
        The particle mass
    d : ndarray, optional
        The total momentum vector of the system.
    L : int, optional
        The lattice size.

    Returns
    -------
    float or ndarray
        The Lorentz boost factor.
    """
    E = EfromMpi(m, np.sqrt(q2), L)
    return WfromE(E, d, L) / E

def calc_Ecm(E, d=np.array([0., 0., 1.]), L=24, lattice=False):
    """Calculates the center of mass energy and the boost factor.

    Calculates the Lorentz boost factor and the center of mass energy
    for moving frames.
    For the lattice dispersion relation see arXiv:1011.5288.

    Parameters
    ----------
    E : float or ndarray
        The energy of the system.
    d : ndarray, optional
        The total momentum vector of the system.
    L : int, optional
        The lattice size.
    lattice : bool
        Use the lattice dispersion relation.

    Returns:
    float or ndarray
        The boost factor.
    float or ndarray
        The center of mass energy.
    """
    # if the data is from the cm system, return immediately
    if np.array_equal(d, np.array([0., 0., 0.])):
        gamma = np.ones_like(E)
        return gamma, E
    if lattice:
        Ecm = EfromW_lat(E, d, L)
    else:
        Ecm = EfromW(E, d, L)
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

def q2fromE_mass_latt(E, m, L=24):
    """Caclulates the q2 from the energy and particle mass
    using the lattice version.

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
    return (np.arcsin(np.sqrt((np.cosh(E*0.5)-np.cosh(m))*0.5))*\
            float(L)/ np.pi)**2

def calc_q2(E, m, L=24, lattice=False):
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
    lattice : bool, optional
        Use the lattice equation.

    Returns
    -------
    float or ndarray
        The CM momentum squared.
    """
    if lattice:
        q2 = q2fromE_mass_latt(E, m, L)
    else:
        q2 = q2fromE_mass(E, m, L)
    return q2
