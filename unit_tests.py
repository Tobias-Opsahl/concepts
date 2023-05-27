import numpy as np


def test_data_generator():
    """
    Try to import data generating functions and create datasets.
    NOTE: MNIST data must be saved in data/. If they are not, please follow the link in
    data_loaders.py in load_mnist() to download (or google MNIST Yann LeCun).
    """
    from data_generator import create_normal_3_5, create_saturated_3_5, create_partial_saturated_3_5
    create_normal_3_5()
    create_saturated_3_5()
    create_partial_saturated_3_5(0.8)
    create_partial_saturated_3_5(0.90)
    create_partial_saturated_3_5(0.95)


def test_load_concept_data():
    """
    Loads the concept data created in data_generator.py, and checks that the data is on the
    correct format. `saturated_3_5.pkl` should have 100% overlap between concept `saturated`
    and the class labels, so check that they are equal for every set.
    """
    from data_loaders import load_concept_data
    # First load without validation set
    (x_train, y_train, c_train), (x_test, y_test, c_test) = \
        load_concept_data("saturated_3_5.pkl", validation_split=False)
    y_train = np.where(y_train == 3, 1, 0)  # Convert labels to 0 and 1, instead of 3 and 5
    y_test = np.where(y_test == 3, 1, 0)
    np.testing.assert_equal(y_train, c_train["saturated"])  # These should be equal, 100% overlap
    np.testing.assert_equal(y_test, c_test["saturated"])
    training_concepts = c_train["saturated"].sum()  # Save amount of concepts to compare to valdiation set
    train_length = x_train.shape[0]

    # Now try with validation set
    (x_train, y_train, c_train), (x_val, y_val, c_val), (x_test, y_test, c_test) = \
        load_concept_data("saturated_3_5.pkl", validation_split=True)
    y_train = np.where(y_train == 3, 1, 0)  # Convert labels to 0 and 1, instead of 3 and 5
    y_val = np.where(y_val == 3, 1, 0)
    y_test = np.where(y_test == 3, 1, 0)
    np.testing.assert_equal(y_train, c_train["saturated"])  # These should be equal, 100% overlap
    np.testing.assert_equal(y_val, c_val["saturated"])
    np.testing.assert_equal(y_test, c_test["saturated"])
    assert (training_concepts) == (c_train["saturated"].sum() + c_val["saturated"].sum())  # Check for same amount
    assert (train_length == x_train.shape[0] + x_val.shape[0])


def test_all():
    """
    Calls all test functions
    """
    test_data_generator()
    test_load_concept_data()


if __name__ == "__main__":
    test_all()
    pass
