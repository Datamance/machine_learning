#!/usr/bin/env/ python
"""Some sick as fuck python utilities my dawg!."""

import csv
import requests
import shutil
import urllib


def download_file(url, filename):
    """Downloads a file, and returns filename."""
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as file:
        shutil.copyfileobj(response, file)
    return filename


def chunks(collection, size):
    """Chunks ya shit generatively, dawg."""
    for i in range(0, len(collection), size):
        yield collection[i: i + size]


def float_range(x, y, step):
    """Slightly broken, has some weird steps."""
    while x < y:
        yield x
        x += step
