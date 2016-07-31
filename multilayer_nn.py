"""A very basic multi-layer Neural Network.

Should be able to support an arbitrary number of layers.

TODO(rrodriguez): Add convolutional capabilities.
"""

import data_utils
import matplotlib.pyplot
import numpy
import operator
import utils
import scipy.special

__author__ = 'Rico Rodriguez'
__email__ = 'rico.l.rodriguez@gmail.com'


# Layer constants.
INPUT = 0
FINAL_LAYER = -1

# URLs.
MNIST_TEST_URL = 'http://pjreddie.com/media/files/mnist_test.csv'
MNIST_TRAINING_URL = 'http://pjreddie.com/media/files/mnist_train.csv'

# Default Data sets.
TRAINING_DATA = data_utils._get_classification_data(MNIST_TRAINING_URL)
TEST_DATA = data_utils._get_classification_data(MNIST_TEST_URL)


class Error(Exception):
    """Base exception for this module."""


class InsufficientLayers(Error):
    """When you don't have enough layers in your ANN."""


# Helper functions.
def _get_weights_for_layers(num_out, num_in):
    """Gets weights between two layers.

    Currently, we just sample weights from a normal probability distribution
    that is centered around 0.0. The standard deviation is related to the
    number of incoming links to a node.
    """
    return numpy.random.normal(0.0, pow(num_out, -0.5), (num_out, num_in))


def _create_connection_matrix(nodedef_seq):
    """Gets the neural network in matrix form."""
    total_length = len(nodedef_seq)
    if total_length < 3:
        raise InsufficientLayers(
            'Need at least 3 layers for NN - input, hidden, and output.')
    matrix = []
    for layer, num_nodes in enumerate(nodedef_seq):
        if layer == total_length - 1:
            # This is our final output layer, so do NOT append weighted axons!
            break
        else:  # connect the dots!
            matrix.append(  # If you had appended, layer + 1 would break!
                _get_weights_for_layers(nodedef_seq[layer + 1], num_nodes))
    return numpy.array(matrix)


class MultiLayerANN(object):
    """Base class for a simple multilayer neural network."""

    def __init__(self, *args, learning_rate=0.08, activation_function=None,
                 training_epochs=7):
        """Constructor."""
        self._nodes_per_layer = numpy.array(args)

        self._weight_matrix = _create_connection_matrix(args)

        self._learning_rate = learning_rate

        # Sigmoid-esque activation function.
        self.activation_function = (
            activation_function or (lambda x: scipy.special.expit(x)))

        # Score is initially set to 0.
        self._score = 0.0

        # Default # training epochs.
        self._epochs = training_epochs

    def _forward_propogate(self, input_column):
        """Propogates an input column through our neural net.

        Returns:
            A list of outputs.
        """
        outputs = []

        for layer in range(len(self._weight_matrix)):
            # 1) Axonal charge gets weighted, that is the output...
            weighted_inputs = numpy.dot(
                self._weight_matrix[layer], input_column)
            # 1) Pulse through neuron layer, activate based on input
            outputs.append(self.activation_function(weighted_inputs))
            # 3) ... which becomes the input for the next iteration.
            input_column = outputs[layer]

        return outputs

    def _backpropogate_errors(self, target_list, outputs, input_column):
        """Backpropogation, using stored outputs to correct weights.

        Remember, outputs are always what comes out of an activation function.
        To become inputs, they must be multiplied by weights at the incoming
        layer.
        """
        # Initialize with a target column that we can start calculating
        # error from. Initialize with final output layer.
        target = numpy.array(target_list, ndmin=2).T
        last_error = target - outputs[FINAL_LAYER]

        # Backstep through outputs.
        for layer in range(len(outputs) - 1, -1, -1):
            output = outputs[layer]

            # Capture error for next layer before correction.
            next_error = numpy.dot(self._weight_matrix[layer].T, last_error)

            # Make sure we have the right input
            inputs = input_column if layer is INPUT else outputs[layer - 1]

            self._weight_matrix[layer] += self._learning_rate * numpy.dot(
                last_error * output * (1.0 - output), numpy.transpose(inputs))

            last_error = next_error

    def _train(self, input_list, target_list):
        """Internal training method for our neural network."""

        input_column = numpy.array(input_list, ndmin=2).T

        outputs = self._forward_propogate(input_column)

        self._backpropogate_errors(target_list, outputs, input_column)

    def train(self, data_url=None, epochs_override=None):
        """Exposed training method, fetches data too."""
        # Fetch training data.
        training_data = (data_utils._get_classification_data(data_url)
                         if data_url else TRAINING_DATA)

        self._epochs = epochs_override or self._epochs

        for epoch in range(self._epochs):
            for record in training_data:
                inputs = numpy.asfarray(record[1:])
                targets = numpy.zeros(
                    self._nodes_per_layer[FINAL_LAYER]) + 0.01
                targets[record[0]] = 0.99
                self._train(inputs, targets)

    def query(self, input_list):
        """Takes the input to our ANN and returns the network's output."""
        return self._forward_propogate(input_list)[FINAL_LAYER]

    def test(self, data_url=None):
        """Tests this Neural Network."""
        test_data = (data_utils._get_classification_data(data_url) if data_url
                     else TEST_DATA)

        scorecard = []

        for test_set in test_data:
            correct_label = test_set[0]
            outputs = self.query(test_set[1:])
            label = numpy.argmax(outputs)
            if label == correct_label:
                scorecard.append(1)
            else:
                scorecard.append(0)

        np_scores = numpy.asarray(scorecard)

        self._score = np_scores.sum() / np_scores.size

        return self._score


def get_tested_network():
    """Tests our Multilayer ANN."""
    neural_network = MultiLayerANN(
        784, 180, 180, 10, learning_rate=0.01, training_epochs=10)

    neural_network.train()
    print(neural_network.test())
    return neural_network

get_tested_network()