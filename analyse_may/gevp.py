import numpy as np
import scipy.linalg as spla

#####
# Everything below coded by Benedikt Sauer
#####

def permutation_indices(data):
    """Sorts the data according to their value.

    This function is called by solve_gevp_gen to sort the eigenvalues
    according to their absolut values. This works on the assumption
    that the eigenvalues are real.

    Parameters
    ----------
    data : ndarray
        The data.

    Returns
    -------
    list of int
        A list where the first entry corresponds to the index of
        largest value, the last entry is corresponds to the index of
        the smallest value.
    """
    return list(reversed(sorted(range(len(data)), key = data.__getitem__)))

def reorder_by_ev(ev1, ev2, B):
    """Creates an index list based on eigenvectors and the matrix B.

    Creates an index list where the first entry corresponds to the
    index of the eigenvector ev2 with largest overlap to the first
    eigenvector of ev1. The last index corresponds to the index of the
    eigenvector ev2 with largest overlap to the last eigenvector ev2
    that did not have a large largest overlap with a previous
    eigenvector ev1.
    WARNING: If more than one eigenvector ev2 has the (numerically)
    same largest overlap with some eigenvector ev1, the behaviour is
    not specified.

    Parameters
    ----------
    ev1 : ndarray
        The eigenvectors to sort by, assumes they are already sorted.
    ev2 : ndarray
        The eigenvectors to sort.
    B : ndarray
        The matrix used during sorting, needed for normalization.

    Returns
    -------
    list
        The indices of the sorted eigenvectors.
    """
    # Calculate all scalar products of the eigenvectors ev1 and ev2. The matrix
    # B is used for the normalization, due to the SciPy eigh solver used. The
    # absolute value is needed because the eigenvectors can also be
    # antiparallel.
    # WARNING: It might be possible that more than one eigenvector ev2 has the
    # (numerically) same largest overlap with some eigenvector ev1. In this
    # case the behaviour is not specified.
    ev1_b = np.dot(np.array(B), ev1)
    dot_products = [ np.abs(np.dot(e, ev1_b)) for e in ev2.transpose() ]
    # Sort the eigenvectors ev2 according to overlap with eigenvectors ev1 by
    # using the scalar product. This assumes that ev1 was already sorted.
    res = []
    # this iterates through the eigenvectors ev1 and looks for the greatest
    # overlap
    for m in dot_products:
        # sort the eigenvectors ev2 according to their overlap with ev1
        for candidate in permutation_indices(m):
            # add each eigenvector ev2 only once to the index list and break
            # when a vector has been added so that only one eigenvector ev2
            # is added for each eigenvector ev1
            if not candidate in res:
                res.append(candidate)
                break
    return res

def solve_gevp_gen(data, t0):
    """Generator that returns the eigenvalues for t_0 -> t where t is in
    (t0, t_max].
       
    Calculate the eigenvalues of the generalised eigenvalue problem
    using the scipy.linalg.eigh solver.
    
    Parameters
    ----------
    a : ndarray
        The time dependent data for the GEVP.
    t0 : int
        The index for the inverted matrix.

    Yields
    ------
    ndarray
        The eigenvalues of the respective time
    ndarray
        The eigenvectors of the respective time
    int
        The time.
    """
    # B is the matrix at t=t0
    B = data[t0]
    # define the eigensystem solver function as a lambda function
    try:
        f = lambda A: spla.eigh(b=B, a=A)
    except LinAlgError:
        return

    # initialization
    eigenvectors = None
    count = 0

    # calculate the eigensystem for t in (t0, T/2+1)
    for j in range(t0 + 1, data.shape[0]):
        try:
            # calculate the eigensystems
            eigenvalues, new_eigenvectors = f(data[j])
            # initialize the new eigenvector array if not done already and
            # sort the eigenvectors and eigenvalues according the absolute
            # value of the eigenvalues
            if eigenvectors is None:
                eigenvectors = np.zeros_like(new_eigenvectors)
                perm = permutation_indices(eigenvalues)
            # Sort the eigensystem by the eigenvectors (for details see the
            # function description). The matrix B is used for the normalization
            # due to the normalization of the eigenvectors return by the eigh
            # solver.
            else:
                perm = reorder_by_ev(new_eigenvectors, eigenvectors, B)
            # permutation of the lists
            eigenvectors = new_eigenvectors[:,perm]
            eigenvalues = eigenvalues[perm]
                
            count += 1

            yield eigenvalues, eigenvectors, j

        except (spla.LinAlgError, TypeError) as e:
            return

#def calculate_gevp(data, t0=1):
def calculate_gevp(data, t0=1):
    """Solves the generalized eigenvalue problem of a correlation
    function matrix.

    The function takes a bootstrapped correlation function matrix and
    calculates the eigenvectors and eigenvalues of the matrix. The
    algorithm relies on the matrix being symmetric or hermitian.

    The eigenvectors are calculated but not stored.

    Parameters
    ----------
    data : ndarray
        The time dependent data for the GEVP.
    t0 : int
        The index for the inverted matrix.

    Returns
    -------
    ndarray
        The array contains the eigenvalues of the solved GEVP. The
        dimension of the array is reduced by one and the data up to t0
        is filled with zeros.
    """

    #Interface to original data format
#    dim = np.int(np.sqrt(data.shape[0]))
#    data = data.T
#    #reshapes matrix elements to actual matrix shape
#    data = np.reshape(data, (data[...,0].shape + (dim, dim)) )
#    data = np.swapaxes(data.T, 2, 3)
#    print data.shape

    values_array = np.zeros(data.shape[:-1])
    if data.ndim == 4:
        # iterate over the bootstrap samples
        for _samples in range(data.shape[0]):
            # iterate over the eigensystems
            for eigenvalues, _eigenvectors, _t in solve_gevp_gen(data[_samples], t0):
                # save the eigenvalues to the array
                values_array[_samples, _t] = eigenvalues
        # set the eigenvalues for t=t0 to 1.0
        values_array[:,t0] = 1.0

    return values_array

#    #Interface to original data format
#    if (data00.shape != data01.shape) or (data00.shape != data11.shape):
#        print("ERROR in calculate_gevp()")
#        print("\tnumber of bootstrap samples or time extent is wrong")
#        os.sysexit(-1)
#
#    # Initialize the eigenvalue array
#    E0 = np.zeros(data00.shape)
#    E1 = np.zeros(data00.shape)
#    for _b in range(0, data00.shape[0]):
#        data = np.empty([data00.shape[1], 2, 2])
#        for _t in range(0, data00.shape[1]):
#            data[_t, 0, 0] = data00[_b, _t]
#            data[_t, 0, 1] = data01[_b, _t]
#            data[_t, 1, 0] = data01[_b, _t]
#            data[_t, 1, 1] = data11[_b, _t]
#
#        # iterate over the bootstrap samples
#        for eigenvalues, _eigenvectors, _t in solve_gevp_gen(data, t0):
#            # save the eigenvalues to the array
#            E0[_b, _t] = eigenvalues[0]
#            E1[_b, _t] = eigenvalues[1]
#        # set the eigenvalues for t=t0 to 1.0
#        #values_array[:,t0] = 1.0
#  #    return values_array
#    return E0, E1









