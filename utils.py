import numpy as np


def check_array(x_array, check_type=True, check_shape=False, check_nans=True, check_inf=True):
    """
    Checks if array satisfies wanted properties.
    If they do not, raises ValueError.

    Arguments:
        x_array (np.array): The array to be checked.
        check_type (bool): Check if array is of type np.ndarray.
        check_shape (int): Checks that the length of the shape of the array is equal to "check_shape".
        check_nans (bool): Check if array contain nans.
        check_inf (bool): Check if array contains inf.
    """
    if check_type and not isinstance(x_array, np.ndarray):
        message = f"Argument must be of type np.ndarray. Was of type {type(x_array)}"
        raise ValueError(message)

    if check_shape and len(x_array.shape) != check_shape:
        message = f"Array expected to have {check_shape} dimensions, but had {len(x_array.shape)}."
        raise ValueError(message)

    if check_nans and np.isnan(x_array).any():
        message = f"Array contains nans."
        raise ValueError(message)

    if check_inf and np.isinf(x_array).any():
        message = f"Array contains nans."
        raise ValueError(message)


def check_arrays(x_array, y_array, dims=[], check_type=True, check_shape=False, check_nans=False, check_inf=False,
                 same_shape_length=False, same_shape=False):
    """
    Check if arrays has the correct shapes and types.

    Arguments:
        x_array (np.array): The first array to be checked.
        y_array (np.array): The second array to be checked.
        dims (list): List of ints of the dimensions that has to match in the arrays.
        check_type (bool): Check if array is of type np.ndarray.
        check_shape (int): Checks that the length of the shape of the array is equal to "check_shape".
        check_nans (bool): Check if array contain nans.
        check_inf (bool): Check if array contains inf.
        same_shape_length (bool): If True, check that arrays have the same amount of dimensions (shape is same length).
        same_shape (bool): If True, check that the arrays match sizes in every dimensions.
    """
    check_array(x_array, check_type, check_shape, check_nans, check_inf)
    check_array(y_array, check_type, check_shape, check_nans, check_inf)

    if same_shape:  # Check that every dimension matches
        same_shape_length = True  # Checks first that we have same amount of dimensions
        dims = np.arange(len(x_array.shape))  # Then set dims to check every dimension.

    if same_shape_length:  # Check that the amount of dimensions are the same.
        if len(x_array.shape) != len(y_array.shape):
            message = f"Array shape-lengths mismatch. "
            message += f"Was of length {len(x_array.shape)} and {len(y_array.shape)}. "
            message += f"Total shape was {x_array.shape} and {y_array.shape}."
            raise ValueError(message)

    for dim in dims:
        if x_array.shape[dim] != y_array.shape[dim]:
            message = f"Array dimenions mismatch. Dimension {dim} must be of same length, "
            message += f"but was {x_array.shape[dim]} and {y_array.shape[dim]}. "
            message += f"Total shape was {x_array.shape} and {y_array.shape}."
            raise ValueError(message)


def integer_one_hot_encode(x_array, max_int=None):
    """
    One hot encodes x_array.
    This assumes that x_arrays only has integer, and that the max element + 1 is the amount of class.
    Therefore, the classes should probably have consequtive values from 0 to c - 1.

    Arguments:
        x_array (np.array): (n) array of values to be one-hot-encoded.
        max_int (int): Max int of class.

    Returns:
        one_hot_array (np.array): (n x c) array of one-hot-encoded data.
    """
    if max_int is None:
        max_int = x_array.max()
    one_hot_array = np.zeros((x_array.shape[0], max_int + 1))  # Initialize empty array
    one_hot_array[np.arange(x_array.shape[0]), x_array] = 1  # Index rows (arange) and columns (x_array)
    one_hot_array = one_hot_array.astype(np.uint8)
    return one_hot_array
