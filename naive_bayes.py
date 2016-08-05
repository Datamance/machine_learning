#!/usr/bin/env python3

"""Classifier for chat data from nltk corpus."""

__author__ = 'Rico Rodriguez'
__credits__ = ['Rico Rodriguez']
__version__ = '0.0.1'
__maintainer__ = 'Rico Rodriguez'
__email__ = 'rico.l.rodriguez@gmail.com'


DIALOGUE_ACT_TYPES = 'DAT'
EMOTICONS = (':)', ':(', ':D', ':P')


import argparse
import matplotlib.pyplot as plt  # TODO(rrodriguez): lazy load this?
import nltk
import numpy
from random import shuffle
from xml.etree.ElementTree import Element


def get_args():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description='Construct NBC.')

    parser.add_argument('-p', '--training_percentage', type=float, default=0.2)
    parser.add_argument('-t', '--test_threshold', type=float, default=0.5)
    parser.add_argument('-k', '--kind', type=str, default=DIALOGUE_ACT_TYPES)
    parser.add_argument(
        '-g', '--graph', type=bool, nargs='?', const=True, default=False)

    args = parser.parse_args()

    # Validate CLI arguments.
    assert 0 < args.training_percentage < 1
    assert 0 < args.test_threshold < 1
    assert args.kind in (DIALOGUE_ACT_TYPES,)  # TODO(rrodriguez): add more later.

    # Could pass more args here eventually.
    return (args.training_percentage, args.test_threshold, args.kind,
            args.graph)


def _get_posts(start, end):
    """Gets a list of posts as XML elements.

    Default behavior is to get all posts.
    """
    posts = list(nltk.corpus.nps_chat.xml_posts()[start:end])
    shuffle(posts)
    return posts


def _get_feature_dict(post):
    """Creates a feature dict for a post.

    TODO(rrodriguez): this feature dict should change contingent on the type of
        classifier we are building.

    TODO(rrodriguez): strengthen features. Be careful not to overfit.
    """
    # Make sure the post is an XML element.
    assert type(post) is Element

    feature_dict = {}
    tokenized_words = nltk.word_tokenize(post.text)
    post_length = len(tokenized_words)

    # Establish all features
    feature_dict['first-word'] = tokenized_words[0]
    feature_dict['ends-with-question'] = tokenized_words[-1] == '?'
    feature_dict['ends-with-exclamation'] = tokenized_words[-1] == '!'
    feature_dict['is-upcase'] = post.text.isupper()

    for word in tokenized_words:
        feature_dict['contains({})'.format(word)] = True

    return feature_dict


def get_feature_sets(start_from=0, up_to=None):
    """Reads XML posts to get proper supervised classification data."""
    return [(_get_feature_dict(post), post.get('class')) for post
            in _get_posts(start_from, up_to)]


def _validate_classifier(classifier, test_set, threshold):
    """Tests the accuracy of a given classifier.

    Limitations:
        1) only tests Naive Bayes classifiers at the moment.
    """
    # Enforce limitations.
    assert type(classifier) is nltk.classify.naivebayes.NaiveBayesClassifier
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print('Test set size: {size}'.format(size=len(test_set)))
    print('Accuracy: {number:.{digits}f}%'.format(number=accuracy * 100, digits=3))
    assert accuracy >= threshold
    print('Threshold of {!s}% passed.'.format(threshold * 100))


def plot_accuracy(data_set):
    """Plots accuracy for our classifier for a single data set that is split.

    TODO(rrodriguez): Make this faster.
    """
    print('PLOTTING ACCURACY...')
    # default to 1% increments
    x_range, y_range = [], []
    for percentage in numpy.arange(0.01, 1, 0.01):
        x_range.append(percentage)
        split_point = int(len(data_set) * percentage)
        training_set, test_set = data_set[:split_point], data_set[split_point:]
        print('training set len: ', len(training_set), '\nTest set len: ', len(test_set))
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        y_range.append(nltk.classify.accuracy(classifier, test_set))

    plt.plot(x_range, y_range)
    plt.title('Classifier Accuracy over single data set')
    plt.ylabel('Accuracy')
    plt.xlabel('Training dataset percentage')
    plt.show()


def get_classifier(training_percentage, test_threshold, of_kind, plot):
    """Classify chats for dialogue.

    This will eventually support P.O.S in addition to dialogue act types.

    Limitations:
        1) Only by dialogue act types at the moment.
        2) NBC construction requires training set at the moment.
    """
    all_data = get_feature_sets()

    # Split point will default to 50% of the total data set.
    split_point = int(len(all_data) * training_percentage)
    training_set, test_set = all_data[:split_point], all_data[split_point:]

    num_training_items = len(training_set)
    print('Training Classifier on {} posts, {:.2f}% of total data set.'.format(
          num_training_items, (num_training_items / len(all_data)) * 100))

    # TODO(rrodriguez): Figure out parameterized construction for other kinds of
    # classifiers.
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print('Training complete.')

    _validate_classifier(classifier, test_set, test_threshold)
    classifier.show_most_informative_features(20)

    if plot:  plot_accuracy(all_data)

    return classifier


def main():
    """Create a classifier based on command line args."""
    return get_classifier(*get_args())


classifier = main()
