import numpy as np
import matplotlib.pyplot as plt
import pickle

from data_loaders import load_mnist, load_fashion_mnist, _load_mnist_generic
from data_visualizer import *


def slice_data(x_data, y_data, digits):
    """
    Slices data so that the returned arrays are only the rows corresponding to where
    the labels `y_data` are in the list `digits`.
    For example, if `digits = [3, 5]`, `x_data` is the mnist_data and `y_data` is the
    mnist labels, the function will return the images and labels correspodning to
    3s and 5s.

    Args:
        x_data (np.array): [n x m] data matrix of features to slice.
        y_data (np.array): [n] array of labels for the x_data, each label is an int.
        digits (list or array): List of ints to of interest.

    Returns:
        x_data (np.array): [n0 x m] data matrix of inputs in `digits`.
        y_data (np.array): [n0] label matrix of inputs in `digits`.
    """
    indices = np.isin(y_data, digits)  # Get boolean of labels in digits
    x_data = x_data[indices]
    y_data = y_data[indices]
    return x_data, y_data


def saturate_images(x_data, indices=None, threshold=127):
    """
    Saturate pixel values in `x_data` that are over `threshold`, that is, sets them to 255.
    If `indices` is not None, will only saturate the images where `indices` is `True`.

    Args:
        x_data (np.array): [n x m] data matrix of features to slice.
        indices (np.array): [n] boolean array of rows to saturate in `x_data`.
        threshold (float): The threshold of where to saturate the pixels.

    Returns:
        x_data (np.array): [n x m] data matrix, where the images of interest are saturated.
    """
    if indices is not None:  # Index only indices of interest
        x_data[indices] = np.where(x_data[indices] > threshold, 255, 0)
    else:
        x_data = np.where(x_data > threshold, 255, 0)  # Set pixels over threshold to be 255
    return x_data


def create_normal_3_5(filename="normal_3_5", read_path="data/", write_path="data/new_data/"):
    """
    Creates a dataset from MNIST, with only 3s and 5s (no saturation).

    Arguments:
        filename (str): The filename of the pickle to save.
        read_path (str): The path to read the (MNIST) data from.
        write_path (str): The path to write the new data to.
    """
    np.random.seed(57)
    (x_train, y_train), (x_test, y_test) = load_mnist(path=read_path, normalize=False, validation_split=False)
    x_train, y_train = slice_data(x_train, y_train, [3, 5])
    x_test, y_test = slice_data(x_test, y_test, [3, 5])
    data_dict = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
                 "concepts_train": {}, "concepts_test": {}}
    with open(write_path + filename + ".pkl", "wb") as outfile:
        pickle.dump(data_dict, outfile)


def create_saturated_3_5(filename="saturated_3_5", read_path="data/", write_path="data/new_data/"):
    """
    Creates a dataset from MNIST, with only 3s and 5s, where the 3s are saturated, but not the 5s.

    Arguments:
        filename (str): The filename of the pickle to save.
        read_path (str): The path to read the (MNIST) data from.
        write_path (str): The path to write the new data to.
    """
    np.random.seed(57)
    (x_train, y_train), (x_test, y_test) = load_mnist(path=read_path, normalize=False, validation_split=False)
    x_train, y_train = slice_data(x_train, y_train, [3, 5])  # Index only 3 and 5
    x_test, y_test = slice_data(x_test, y_test, [3, 5])
    train_indices = np.isin(y_train, [3])  # Find indices where saturation will be performed
    test_indices = np.isin(y_test, [3])
    x_train = saturate_images(x_train, train_indices, threshold=127)  # DO the saturation
    x_test = saturate_images(x_test, test_indices, threshold=127)
    train_indices = train_indices.astype(int)  # Convert indices to int for the dataset
    test_indices = test_indices.astype(int)
    data_dict = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
                 "concepts_train": {"saturated": train_indices}, "concepts_test": {"saturated": test_indices}}
    with open(write_path + filename + ".pkl", "wb") as outfile:  # Write the dataset
        pickle.dump(data_dict, outfile)


def create_partial_saturated_3_5(split=0.8, filename="partial", read_path="data/", write_path="data/new_data/"):
    """
    Creates a dataset from MNIST, with only 3s and 5s, where `split`-ratio of the 3 digits are saturated, and
    1 - `split` of the 5 indices are saturated.

    Arguments:
        split (float): Float between 0 and 1 that determined the ratio of saturated 3 digits,
            and 1 - `split` saturated 5 digits.
        filename (str): The filename of the pickle to save.
        read_path (str): The path to read the (MNIST) data from.
        write_path (str): The path to write the new data to.
    """
    np.random.seed(57)  # Set seed
    (x_train, y_train), (x_test, y_test) = load_mnist(path=read_path, normalize=False, validation_split=False)
    x_train, y_train = slice_data(x_train, y_train, [3, 5])  # Index only the 3 and 5s
    x_test, y_test = slice_data(x_test, y_test, [3, 5])
    # First mark every digit that is a 3:
    train_indices = np.isin(y_train, [3])
    test_indices = np.isin(y_test, [3])
    # Then randomly flip 20% of the indices to create some noise:
    n_train = len(train_indices)
    n_test = len(test_indices)
    flip_indices_train = np.random.choice(n_train, int(n_train * (1 - split)), replace=False)  # Indices to flip
    flip_indices_test = np.random.choice(n_test, int(n_test * (1 - split)), replace=False)
    train_indices[flip_indices_train] = np.logical_not(train_indices[flip_indices_train])  # Flip the indices
    test_indices[flip_indices_test] = np.logical_not(test_indices[flip_indices_test])
    x_train = saturate_images(x_train, train_indices, threshold=127)  # Saturate the indices found
    x_test = saturate_images(x_test, test_indices, threshold=127)
    train_indices = train_indices.astype(int)  # Convert indices to int for the dataset
    test_indices = test_indices.astype(int)
    data_dict = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
                 "concepts_train": {"saturated": train_indices}, "concepts_test": {"saturated": test_indices}}
    total_filename = write_path + filename + str(int(split * 100)) + "_3_5.pkl"
    with open(total_filename, "wb") as outfile:  # Write the dataset
        pickle.dump(data_dict, outfile)


if __name__ == "__main__":
    create_normal_3_5()
    create_saturated_3_5()
    create_partial_saturated_3_5(0.8)
    create_partial_saturated_3_5(0.90)
    create_partial_saturated_3_5(0.95)
    # from IPython import embed
    # embed()
