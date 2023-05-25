import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class CAV:
    """
    Concept Activation Vectors (CAV)
    """
    def __init__(self, model, layer_names, concept_names=["concept1"], hparams=None):
        self.model = model
        self.layer_names = layer_names
        self.layers = [model.get_layer(layer_name) for layer_name in layer_names]
        self.models = []  # Store intermediary models in a list
        for layer in self.layers:
            intermediary_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
            self.models.append(intermediary_model)
        self.concept_names = concept_names
        self._set_hparams(hparams)  # Set hyperparameters

    def train(self, inputs, concept_labels, concept_name="concept1"):
        # if not isinstance(inputs, dict):  # In case inputs is not dict and we only have one concept
        #     inputs = {self.concept_names[0]: inputs}

        activations = self._get_activations(inputs)
        linear_model = LogisticRegression()
        x_train, x_test, y_train, y_test = self._preprocess(activations, concept_labels)
        linear_model.fit(x_train, y_train)
        preds = linear_model.predict(x_test)
        accuracy = accuracy_score(preds, y_test)
        return accuracy

    def _set_hparams(self, hparams):
        if hparams is None:
            self.hparams = self._default_hparams()
        # TODO: Check that the hyperparameter dict contains the correct stuff and is dict
        self.hparams = hparams

    def _default_hparams(self):
        # TODO: Set default hyperparams
        return {}

    def _preprocess(self, inputs, concept_labels):
        # TODO: Make activations correctly
        # activations = inputs.reshape(inputs.shape[0], -1)
        x_train, x_test, y_train, y_test = train_test_split(inputs, concept_labels,
                                                            test_size=0.33, stratify=concept_labels, random_state=57)
        return x_train, x_test, y_train, y_test

    def _get_activations(self, inputs):
        activations = {}
        for i in range(len(self.models)):
            activations[self.layer_names[i]] = self.models[i].predict(inputs)
        return activations
