"""A very basic single-layer Neural Network."""

import data_utils
import matplotlib.pyplot
import numpy
import operator
import utils
import requests
import scipy.special

__author__ = 'Rico Rodriguez'
__email__ = 'rico.l.rodriguez@gmail.com'


# Constants.
INPUT = 'input'
INNER = HIDDEN = 'hidden'
OUTPUT = 'output'
ITH = 'input_to_hidden'
HTO = 'hidden_to_output'

# Default test boundaries.
DEFAULT_HIDDEN_BOUNDS = (180, 220)
DEFAULT_FLOAT_RANGE = (0.02, 0.15, 0.01)
DEFAULT_EPOCH_BOUNDS = (2, 10)

# URLs.
MNIST_TEST_URL = 'http://pjreddie.com/media/files/mnist_test.csv'
MNIST_TRAINING_URL = 'http://pjreddie.com/media/files/mnist_train.csv'

# Default Data sets.
TRAINING_DATA = data_utils._get_classification_data(MNIST_TRAINING_URL)
TEST_DATA = data_utils._get_classification_data(MNIST_TEST_URL)


class NeuralNetwork(object):
    """Base class for a simple neural network."""

    def __init__(self, num_inputs, num_inners, num_outputs,
                 learning_rate, activation_function=None):
        """Constructor."""
        self._num_nodes = {
            INPUT: num_inputs,
            INNER: num_inners,
            OUTPUT: num_outputs
        }

        self._learning_rate = learning_rate

        # Sample weights from normal probability distribution centered around
        # zero, with a standard deviation related to the number of incoming
        # links into a node.
        self.weights = {
            ITH: numpy.random.normal(
                0.0, pow(self._num_nodes[INNER], -0.5),
                (self._num_nodes[INNER], self._num_nodes[INPUT])),
            HTO: numpy.random.normal(
                0.0, pow(self._num_nodes[OUTPUT], -0.5),
                (self._num_nodes[OUTPUT], self._num_nodes[INNER]))
        }

        # Sigmoid-esque activation function.
        self.activation_function = (
            activation_function or (lambda x: scipy.special.expit(x)))

        # Score is initially set to 0.
        self._score = 0.0

        # Default # training epochs.
        self._training_epochs = 7

    def train(self, data_url=None, epochs_override=None):
        """Exposed training method, fetches data too."""
        # Fetch training data.
        training_data = (_get_classification_data(data_url) if data_url
                         else TRAINING_DATA)

        self._training_epochs = epochs_override or self._training_epochs

        for epoch in range(self._training_epochs):
            for record in training_data:
                inputs = numpy.asfarray(record[1:])
                targets = numpy.zeros(self._num_nodes[OUTPUT]) + 0.01
                targets[record[0]] = 0.99
                self._train(inputs, targets)

    def _train(self, input_list, target_list):
        """Internal training method for our neural network."""
        # 1) Get transposed arrays - one column.
        inputs = numpy.array(input_list, ndmin=2).T

        # 2) Forward propogation.
        hidden_inputs = numpy.dot(self.weights[ITH], inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weights[HTO], hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 3) Get error.
        targets = numpy.array(target_list, ndmin=2).T
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.weights[HTO].T,
                                  output_errors)

        self.weights[HTO] += self._learning_rate * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs))

        self.weights[ITH] += self._learning_rate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs))

    def query(self, input_list):
        """Takes the input to our ANN and returns the network's output."""
        # Get transposed inputs.
        inputs = numpy.array(input_list, ndmin=2).T

        # Calculate signals in hidden layers.
        hidden_inputs = numpy.dot(self.weights[ITH], inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into a final output layer
        final_inputs = numpy.dot(self.weights[HTO], hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def test(self, data_url=None):
        """Tests this Neural Network."""
        test_data = (_get_classification_data(data_url) if data_url
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

    def __str__(self):
        return ('Neural Network: HIDDEN: {!s}, L-RATE: {!s}, EPOCHS: {!s}, '
                'SCORE: {!s}').format(
            self._num_nodes[HIDDEN], self._learning_rate,
            self._training_epochs, self._score)


def get_tested_network(num_inputs=784, num_hidden=200, num_outputs=10,
                       learning_rate=0.08, num_epochs=7):
    """Tests our neural network."""
    neural_network = NeuralNetwork(
        num_inputs, num_hidden, num_outputs, learning_rate)

    neural_network.train(epochs_override=num_epochs)
    print(neural_network.test())
    print(neural_network)
    return neural_network


def get_candidates(hidden_bounds=DEFAULT_HIDDEN_BOUNDS,
                   float_range_args=DEFAULT_FLOAT_RANGE,
                   epoch_bounds=DEFAULT_EPOCH_BOUNDS):
    """Gets all the candidates for an analysis.

    TODO(rico): Find out how to chunk this work.
        ProcessPoolExecutor barfs, probably because of pipe size.
    """
    candidates = []

    for num_hidden in range(*hidden_bounds):
        for rate in utils.float_range(*float_range_args):
            for num_epochs in range(*epoch_bounds):
                candidates.append(
                    get_tested_network(num_hidden=num_hidden,
                                       learning_rate=rate,
                                       num_epochs=num_epochs
                                       ))

    return candidates


def find_best():
    """Finds the best amongst candidates."""

    candidates = get_candidates()

    winner = max(candidates, key=operator.attrgetter('_score'))

    print('WE HAVE A WINNER!')
    print(winner)


get_tested_network()
# find_best()
