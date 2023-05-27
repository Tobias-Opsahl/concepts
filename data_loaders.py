import numpy as np
import pickle
import gzip
from sklearn.model_selection import train_test_split


def load_mnist(path="data/", normalize=False, validation_split=False):
    """
    https://yann.lecun.com/exdb/mnist/ (do not open in chrome for some reason)
    Load the MNIST dataset. It should be saved as mnist.pkl.gz in the same folder.

    Arguments:
        path (str): Path to enter the folder where the data is store, relative to
            where this function is called.
        normalize (bool): If True, scales pixels down to be between [0, 1].
        validation_split (bool): If True, will split the training set of 60000 images into a training
            set of 50000 images and a validation set of 10000 images.

    Returns:
        training_data (tuple): Tuple consisting of (x_train, y_train), where x_train [50000 x 784]
            is the array of all the training images, and y_train [50000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        validation_data (tuple): Tuple consisting of (x_val, y_val), where x_val [10000 x 784]
            is the array of all the validation images, and y_val [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        test_data (tuple): Tuple consisting of (x_test, y_test), where x_test [10000 x 784]
            is the array of all the testing images, and y_test [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
    """
    return _load_mnist_generic("mnist", path, normalize, validation_split)


def load_fashion_mnist(path="data/", normalize=False, validation_split=False):
    """
    https://github.com/zalandoresearch/fashion-mnist
    Load the fashion MNIST dataset. It should be saved as mnist.pkl.gz in the same folder.
    The data is also split with a validation-set, taking 10000 of the 60000 train pictures
    into a validation set, making it on the same format as the original MNIST dataset.

    Arguments:
        path (str): Path to enter the folder where the data is store, relative to
            where this function is called.
        normalize (bool): If True, scales pixels down to be between [0, 1].
        validation_split (bool): If True, will split the training set of 60000 images into a training
            set of 50000 images and a validation set of 10000 images.

    Returns:
        training_data (tuple): Tuple consisting of (x_train, y_train), where x_train [50000 x 784]
            is the array of all the training images, and y_train [50000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        validation_data (tuple): Tuple consisting of (x_val, y_val), where x_val [10000 x 784]
            is the array of all the validation images, and y_val [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        test_data (tuple): Tuple consisting of (x_test, y_test), where x_test [10000 x 784]
            is the array of all the testing images, and y_test [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
    """
    return _load_mnist_generic("fashion-mnist", path, normalize, validation_split)


def _load_mnist_generic(dataset, path="data/", normalize=False, validation_split=False):
    """
    Generic function for reading either the original MNIST dataset of fashion-MNIST dataset.

    Arguments:
        dataset (str): The dataset to read, either "mnist" or "fashion-mnist".
        path (str): Path to enter the folder where the data is store, relative to
            where this function is called.
        normalize (bool): If True, scales pixels down to be between [0, 1].
        validation_split (bool): If True, will split the training set of 60000 images into a training
            set of 50000 images and a validation set of 10000 images.

    Returns:
        training_data (tuple): Tuple consisting of (x_train, y_train), where x_train [50000 x 784] or [60000 x 784]
            is the array of all the training images, and y_train [50000] or [60000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        (optional) validation_data (tuple): Tuple consisting of (x_val, y_val), where x_val [10000 x 784]
            is the array of all the validation images, and y_val [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        test_data (tuple): Tuple consisting of (x_test, y_test), where x_test [10000 x 784]
            is the array of all the testing images, and y_test [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
    """
    with gzip.open(path + dataset + "/train-labels-idx1-ubyte.gz", "rb") as infile:
        train_labels = np.frombuffer(infile.read(), dtype=np.uint8, offset=8)
    n_train = len(train_labels)
    with gzip.open(path + dataset + "/train-images-idx3-ubyte.gz", "rb") as infile:
        train_data = np.frombuffer(infile.read(), dtype=np.uint8, offset=16).reshape(n_train, 784)

    with gzip.open(path + dataset + "/t10k-labels-idx1-ubyte.gz", "rb") as infile:
        test_labels = np.frombuffer(infile.read(), dtype=np.uint8, offset=8)
    n_test = len(test_labels)
    with gzip.open(path + dataset + "/t10k-images-idx3-ubyte.gz", "rb") as infile:
        test_data = np.frombuffer(infile.read(), dtype=np.uint8, offset=16).reshape(n_test, 784)

    if normalize:  # Pixels are between 0 and 255, so simply dividing by 255 turn pixels into [0, 1]
        train_data = train_data / 255
        test_data = test_data / 255

    if validation_split:
        x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=10000, random_state=57)
        x_test, y_test = test_data, test_labels
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    else:  # No valdidation set, return train and test
        return (train_data, train_labels), (test_data, test_labels)


def load_concept_data(filename, path="data/new_data/", normalize=False, reshape=False,
                      validation_split=False, val_size=0.2):
    """
    Loads and returns one of the concepts dataset in new_data.
    This consits of a train, optinal validation, and test set, with both the x_data, y_data and concepts-dictionary.
    The concept dictionary is a dictionary where the value contains the concept labels for a given concept key.

    Arguments:
        filename (str): Name of dataset to load. Must be a pickle file and contain the ".pkl".
        path (str, optional): Path to the folder where the data is stored. Defaults to "data/new_data/".
        normalize (bool, optional): If True, scales pixels down to be between [0, 1].
        reshape (bool, optional): If True, reshapes from [n, 784] to [n, 28, 28, 1].
        validation_split (bool, optional): If True, will split the training set into train and validation set.
            This means that both the x_data, y_data, and all of the concepts will be split. The proportion
            is determined by `val_size`.
        val_size (float): The ratio of elements that gets put into the valdiation set if `validation_split` is True.

    Returns:
        training_data (tuple): Tuple consisting of (x_train, y_train, concepts_train).
            x_train is shape [batch_size, dim_size], y_train is shape [batch_size],
            concepts_train: {"concept1": concept_array1, "concept2: concept_array2, ...},
            where each concept_array is a binary array (0 or 1) of dim 1 of size [batch_size].
            If there are no concepts, it will be an empty dictionary.
        (optional) validation_data (tuple): Tuple consisting of (x_val, y_val, concepts_val).
            The shapes are described in `training_data` above.
        test_data (tuple): Tuple consisting of (x_test, y_test, concepts_test).
            The shapes are described in `training_data` above.
    """
    with open(path + filename, "rb") as infile:  # Open file and read stuff
        data_dict = pickle.load(infile)
        x_train = data_dict["x_train"]
        y_train = data_dict["y_train"]
        x_test = data_dict["x_test"]
        y_test = data_dict["y_test"]
        concepts_train = data_dict["concepts_train"]
        concepts_test = data_dict["concepts_test"]

    if normalize:  # Pixels are between 0 and 255, so simply dividing by 255 turn pixels into [0, 1]
        x_test = x_test / 255
        x_test = x_test / 255

    if reshape:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

    if not validation_split:  # Return only train and test
        return (x_train, y_train, concepts_train), (x_test, y_test, concepts_test)

    # `validation_split` = True, split train into train and validation
    shuffle_indices = np.random.choice(len(x_train), len(x_train), replace=False)   # Put indices into random order
    # Shuffle all of the data according to the indices
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    for concept in concepts_train:
        concepts_train[concept] = concepts_train[concept][shuffle_indices]

    # Index the train and validation sets
    split_point = int(len(x_train) * (1 - val_size))
    x_train, x_val = x_train[:split_point], x_train[split_point:]
    y_train, y_val = y_train[:split_point], y_train[split_point:]
    concepts_val = {}  # Make new concept dictionary for validation set
    for concept in concepts_train:
        concept_vector = concepts_train[concept]  # The concept vector
        concepts_train[concept], concepts_val[concept] = concept_vector[:split_point], concept_vector[split_point:]

    return (x_train, y_train, concepts_train), (x_val, y_val, concepts_val), (x_test, y_test, concepts_test)


if __name__ == "__main__":
    pass
