import requests


def _scale(value, lower_bound=0.99, upper_bound=255.0):
    """Scales the input to between 0 and 1.

    Defaults to grayscale.

    :param: value
    :param: lower_bound
    :param: upper_bound
    """
    return (value / upper_bound * lower_bound) + (1 - lower_bound)


def _get_classification_data(url, **kwargs):
    """Gets the classification data from a given url"""
    data_array = []

    for line in requests.get(url).text.split('\n')[:-1]:  # Empty last element.
        string_array = line.split(',')
        data_array.append(
            [int(string_array[0])] +
            [_scale(float(num), **kwargs) for num in string_array[1:]])

    return data_array
