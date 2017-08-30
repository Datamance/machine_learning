# -*- coding: utf-8 -*-
"""
Linear Regression from scratch.

This is not meant to be a performant implementation of Linear Regression,
rather it is intended to be as simple as possible so as to be understood
in terms of the serial steps needed to do the actual math.
"""

from random import seed, randrange
from csv import reader
from math import sqrt


def load_csv(filename):
    """Loads the data in list form."""
    dataset = []
    with open(filename, 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            if row:
                dataset.append(row)
            else:
                continue
    return dataset


def str_column_to_float(dataset, column):
    """Converts string column to float."""
    for row in dataset:
        row[column] = float(row[column].strip())


def train_test_split(dataset, split):
    """"Simple cross-validation split.

    TODO(Datamance):
        - K-folds
    """
    training = []
    train_size = split * len(dataset)
    dataset_copy = list(dataset)

    # Randomly sample for training - don't just pick the same ones every time.
    while len(training) < train_size:
        index = randrange(len(dataset_copy))
        training.append(dataset_copy.pop(index))
    
    return training, dataset_copy


def get_rmse(actual, predicted):
    """Get Root Mean Squared Error."""
    error_sum = 0.0
    
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        error_sum += (prediction_error ** 2)
        
    mean_squared_error = error_sum / float(len(actual))
    
    return sqrt(mean_squared_error)


def evaluate_algorithm(dataset, algorithm, split, *args):
    """Evaluates the efficacy of our linear model.
    
    TODO(Datamance):
        - Extend to support different forms of validation
        - Test multiple algorithm types
    """
    train, test = train_test_split(dataset, split)
    test_set = []

    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)

    predicted = algorithm(train, test_set, *args)
    
    actual = [row[-1] for row in test]
    rmse = get_rmse(actual, predicted)
    
    return rmse


def mean(values):
    return sum(values) / float(len(values))


def covariance(x, mean_x, y, mean_y):
    """Calculate covariance.

    Nota Bene: when determining the b0 (slope) coefficient, we are using
    covariance as the "rise"/y factor. If we're summing up all the products
    of the mean-differences for x's and y's, then you can see how this works:
    when both differences are negative, they multiply by each other to add
    a positive number to the covariance sum. When both are positive, the same
    thing happens.
    
    This means, when the signs match, the total goes up. And the larger the
    magnitude within the signs, the more the covariance total goes up.
    """
    result = 0.0
    for i in range(len(x)):
        result += (x[i] - mean_x) * (y[i] - mean_y)
    return result


def variance(values, mean):
    return sum([(x - mean) ** 2 for x in values])


def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    
    x_mean, y_mean = mean(x), mean(y)

    # Slope is covariance(x,y) divided by variance in x.
    # Note here that covariance multiplies x-variation away from the x-mean by
    # y-variation away from the y-mean, while variance is just squaring the
    # variations for x (before summing them). So we are treating covariance
    # like "rise" and variance like "run."
    slope = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    
    y_intercept = y_mean - slope * x_mean
    
    return (y_intercept, slope)


def simple_linear_regression(train, test):
    predictions = []
    y_intercept, slope = coefficients(train)
    
    for row in test:
        predictions.append(y_intercept + (slope * row[0]))
    
    return predictions


# OK, now let's test on the insurance dataset
seed(1)

dataset = load_csv('insurance.csv')

for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)


# Choosing a 0.6 split.
rmse = evaluate_algorithm(dataset, simple_linear_regression, 0.6)

print('RMSE: %.3f' % (rmse,))
