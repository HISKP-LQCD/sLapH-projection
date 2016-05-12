"""
memoize wrapper for functions.
"""

import cPickle

def memoize(function, limit=None):
    """Function decorator for caching results.
    
    The caching is implemented as a dictionary, the key is created from the 
    arguments to the function. Only any 'verbose' flag is skipped.
    The storage implements a LRU cache, if the size is limited. The least used
    element is dropped when the limit is reached.

    Parameters
    ----------
    function: callable
        The function to wrap.
    limit: int
        The maximum number of results cached.

    Returns
    -------
        The decorated function.
    """
    # return immediately if the function has no arguments
    if isinstance(function, int):
        def memoize_wrapper(f):
            return memoize(f, function)

        return memoize_wrapper
    # create the dictionary and a list of keys
    dict = {}
    list = []
    def memoize_wrapper(*args, **kwargs):
        # filter arguments that should not be hashed
        kwa = kwargs
        if 'verbose' in kwa:
            kwa.pop('verbose')
        key = cPickle.dumps((args, kwa))
        try:
            # see if key is in list and if so append it to the end
            list.append(list.pop(list.index(key)))
        except ValueError:
            # if key is not in list, create it
            dict[key] = function(*args, **kwargs)
            list.append(key)
            # if size is limited and the limit is reached, delete first element
            if limit is not None and len(list) > limit:
                del dict[list.pop(0)]
        return dict[key]
    # save the list and the dictionary to the function
    memoize_wrapper._memoize_dict = dict
    memoize_wrapper._memoize_list = list
    memoize_wrapper._memoize_limit = limit
    memoize_wrapper._memoize_origfunc = function
    memoize_wrapper.func_name = function.func_name
    return memoize_wrapper

