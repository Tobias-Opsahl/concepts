import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import check_arrays


class CAV:
    """
    Concept Activation Vectors (CAV)
    https://arxiv.org/abs/1711.11279

    Class for finding the amount of concept activation for a given model and given concepts and
    bottleneck layer. A concept is a binary feature, and to check wether or not cares about a concept,
    a linear model is fitted in some (bottleneck) layer of the model. If the linear model has a high
    accuracy, we assume that the model cares about the concept to a certain extent.
    More precisely, input of both positives and negatives to the class are given, with their label.
    Then, their activation of a layer are calculated, which is the output of that layer with the inputs.
    The linear model gets these activation as input, and the concept labels as targets.
    """
    def __init__(self, model, layer_names, hparams=None):
        """
        Sets class variables

        Args:
            model (model): The model to be used. Can be any model, but may need a wrapper. Needs to
                implement `model.get_layer(layername)` and `model.predict(inputs)`. This is already
                done in a tensorflor.keras model, which is default since the CAV paper was by google.
            layer_names (list of str): List of layer names that we want to check for concepts in.
                `model.get_layer(layer_names[i])` must return the output of the layer.
            hparams (dict, optional): Hyperparameters for the linear model. If None, will be set to default params.
        """
        # Check that model is on correct form
        if not hasattr(model, "get_layer") or not hasattr(model, "predict"):
            message = "Argument `model` to `CAV.__init__()` must implement methods `get_layer` and `predict`."
            raise ValueError(message)

        self.model = model  # Save the class variables
        self.layer_names = layer_names
        layers = [model.get_layer(layer_name) for layer_name in layer_names]  # Access the actual layers
        self.models = []  # Store intermediary models in a list
        for layer in layers:  # Make intermediate models from inputs to bottleneck layers
            intermediary_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
            self.models.append(intermediary_model)
        self._set_hparams(hparams)  # Set hyperparameters

    def train(self, inputs, concept_labels, concept_names=None):
        """
        Trains the linear model to check for the concept. The input is a dict of inputs for the model
        with corresponding binary concept labels.

        Args:
            inputs (np.array): [n x m] array of n input vectors with m features. This must correspond to the
                input shape of the model given in the constructor. m may represent a tuple.
            concept_labels (dict of np.array): Dict of [n] array of binary concept labels corresponding to `inputs`.
                For example on the form {"striped": [0, 1, 1, 0, ...], "zigzag": [1, 1, 0, 1,], ... }.
                The keys will be used for the output. If `concept_lables` is a [n] shaped array, will be converted to
                {"concept1": concept_lables}.
            concept_names (list of str, optional): List of the names of the concepts to check for. If None, will
                use all of the keys in `concept_lables`. Use this argument if one only want to check for some of the
                concepts that are present in `concept_labels`.

        Returns:
            accuracies (dict): Dictionaries of the test / validation accuracies for the linear model.
                TODO: Determine exact format.
        """
        # Check type of inputs and convert if necessary
        if isinstance(concept_labels, (list, np.ndarray)):  # If array, assume only one concept.
            concept_labels = {"concept1": concept_labels}
        if not isinstance(concept_labels, (dict)):
            message = "Argument `concept_labels` to `CAV.train()` must be dictionary of key: value pairs "
            message += "`concept_name: concept_labels` where concept_lables is a label array of 0 and 1. Was of "
            message += f" type {type(concept_labels)}"
            raise TypeError(message)
        for concept in concept_labels:  # Check that objects are arrays and first dimensions are of same size
            check_arrays(inputs, concept_labels[concept], dims=[0])
        if concept_names is None:  # If none, Set to all of the keys in the concept_lables dict
            concept_names = list(concept_labels.keys())
        # lm = linear_model.SGDClassifier(alpha=self.hparams['alpha'], max_iter=self.hparams['max_iter'],
        #                                 tol=self.hparams['tol'])

        activations = self._get_activations(inputs)
        linear_model = LogisticRegression(max_iter=self.hparams["max_iter"])
        accuracies = {}
        for layer in self.layer_names:
            accuracies[layer] = {}
            for concept in concept_names:
                x_data = activations[layer]  # Get correct activations
                y_data = concept_labels[concept]  # Get correct concept labels
                x_train, x_test, y_train, y_test = self._preprocess(x_data, y_data)  # Flatten and split data
                linear_model.fit(x_train, y_train)  # Train, then predict and measure accuracy
                preds = linear_model.predict(x_test)
                accuracy = accuracy_score(preds, y_test)
                accuracies[layer][concept] = accuracy  # Store results
        return accuracies

    def _set_hparams(self, hparams):
        """
        Sets the hyperparameters for the linear model.

        Args:
            hparams (dict): Dict of hyperparameters. TODO: Determine valid key formats.
        """
        if hparams is None:
            hparams = self._default_hparams()
        self.hparams = hparams

    def _default_hparams(self):
        """
        Returns the default hyperparameters for the linear model

        Returns:
            hparams: Dictionary of default hyperparameters
        """
        return {"max_iter": 3000}

    def _preprocess(self, x_data, y_data, test_size=0.33):
        """
        Preprocesses the sets used for training the linear concept model.
        This means splitting inputs and labels in training and test set, and flattening the x_data.
        This preprocess *one* array / tensor of data at a time. This means `x_data` and `y_data` need to be
        array corresponding to *one* bottleneck layer and concept, not a dictionary of many.

        Args:
            x_data (np.array): [n x m] array corresponding to activations in a certain layer. `m` might be a tuple,
                will be flattened.
            y_data (_type_): [n] array of binary concept labels corresponing to the input data made to calculate
                the activations `x_data`.
            test_size (float or int): The proportion of test data of float, or total number of test data if int.

        Returns:
            x_train (np.array): Flattened array of train inputs.
            x_test (np.array): Flattened array of test input.
            y_train (np.array): Labels for train, inputs.
            y_test (np.array): Labels for test inputs.
        """
        x_data = x_data.reshape(x_data.shape[0], -1)  # Flatten array
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                            test_size=test_size, stratify=y_data, random_state=57)
        return x_train, x_test, y_train, y_test

    def _get_activations(self, inputs):
        """
        Caclulate the activations in the layers of interest.
        The layers used as input for the constructer is used, and this method returns a dictionary of
        layer names as keys and the activations array or tensors as values.

        Args:
            inputs (np.array): [n, m] np.array of n inputs with m features. m might be a tuple.

        Returns:
            activations (dict): Dictionary of the activations calculated, on the form
                {layer_name1: activations1, layer_names2: activations2, ... }.
        """
        activations = {}
        for i in range(len(self.models)):  # Loop over the intermediary models made in the constructor
            activations[self.layer_names[i]] = self.models[i].predict(inputs, batch_size=128)  # Predict on input
        return activations
