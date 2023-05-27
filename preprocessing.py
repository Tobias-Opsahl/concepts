import numpy as np
import tensorflow as tf


def preprocess_concept_data(x_data, y_data, normalize=False):
    """
    Preprocesses the concept data, see data_loaders.load_concept_data().
    Takes in ones set, that means, either train, validation or test, one at a time

    Args:
        x_data (np.array): [n x 784] array of images
        y_data (np.array): [n] array of labels
        normalize (bool, optional): If True, scales pixels down to be between [0, 1].
    """
    if normalize:  # Pixels are between 0 and 255, so simply dividing by 255 turn pixels into [0, 1]
        train_data = train_data / 255
        test_data = test_data / 255

    x_data = x_data.reshape(-1, 28, 28, 1)  # Reshape x into image
    y_data = np.where(y_data == 3, 1, 0)  # Make 3s and 5s into 0 and 1s.
    return x_data, y_data


def numpy_to_tfds(x_data, y_data, img_height=28, img_widght=28, n_channels=1, shuffle_size=1000, batch_size=32):
    """
    Creates a TensorFlowDataset (tfds) out of a numpy data array and labels

    Args:
        x_data (np.array): [n x m] array of input data
        y_data (np.array): [n] array of input labels

    Returns:
        dataset (tf.Dataset) tfds object with x_data and y_data provided.
    """
    y_data = np.where(y_data == 3, 1, 0)  # Convert labels to 0 and 1, instead of 3 and 5
    x_data = x_data.reshape(-1, img_height, img_widght, n_channels)  # Reshape from flat shape to image with channels

    # Transform to tfds datatype
    train_ds = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(shuffle_size).batch(batch_size)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_ds.cache().prefetch(buffer_size=AUTOTUNE)  # Cache and prefetch
    return train_dataset


def tfds_to_numpy(dataset):
    """
    Creates a numpy input array and numpy label array from a TensorFlowDataset (tfds) object.


    Arguments:
        dataset (tf.Dataset) tfds object, with both input data and labels.

    Returns:
        x_data (np.array): [n x m] array of input data
        y_data (np.array): [n] array of input labels
    """
    pass


if __name__ == "__main__":
    pass
