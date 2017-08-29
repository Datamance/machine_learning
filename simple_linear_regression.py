# -*- coding: utf-8 -*-
"""
Linear Regression from scratch.

This is not meant to be a performant implementation of Linear Regression,
rather it is intended to be as simple as possible so as to have instructional
value.
"""

from random import seed, randrange
from csv import reader
from math import sqrt


def load_csv(filename):
    """Loads the data"""
    dataset = []
    with open(filename, 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_column_to_float(dataset, column):
    """Converts string column to float."""
    for row in dataset:
        row[column] = float(row[column].strip())


def train_test_split(dataset, split):
    """"Splits a dataset into a training set and a test set."""
    training = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)

    while len(training) < train_size:
        index = randrange(len(dataset_copy))
        training.append(dataset_copy.pop(index))
    
    return training, dataset_copy


def get_rmse(actual, predicted):
    """Get Root Mean Squared Error."""
    sum_error = 0.0
    
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += prediction_error ** 2
        
    mean_error = sum_error / float(len(actual))
    
    return sqrt(mean_error)


def evaluate_algorithm(dataset, algorithm, split, *args):
    """Evaluates how close we got with the minimization."""
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
    
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    
    b0 = y_mean - b1 * x_mean
    
    return (b0, b1)


def simple_linear_regression(train, test):
    predictions = []
    b0, b1 = coefficients(train)
    
    for row in test:
        y_hat = b0 + b1 * row[0]
        predictions.append(y_hat)
    
    return predictions


# OK, now let's test on the insurance dataset
seed(1)

dataset = load_csv('insurance.csv')

for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)


# Choosing a 0.6 split.
rmse = evaluate_algorithm(dataset, simple_linear_regression, 0.6)

print('RMSE: %.3f' % (rmse,))
