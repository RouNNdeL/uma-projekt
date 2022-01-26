import csv

import numpy as np


def load_data_from_csv(path, delimiter=","):
    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        next(csv_reader)
        data = []
        for row in csv_reader:
            data.append(row)
        return format_data(np.array(data, dtype=np.float64))


def format_data(data):
    # Format data, moving last value to the first position
    data = np.roll(data, 1, axis=1)
    return data
