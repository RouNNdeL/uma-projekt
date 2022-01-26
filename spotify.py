#!/usr/bin/env python3
# dataset: https://www.kaggle.com/lehaknarnauli/spotify-datasets
import json
import time

import numpy
import numpy as np

from regtree.tree import RandomForest
from utils import load_data_from_csv


def check(
        train,
        test,
        feedback: bool,
        trees: int,
        samples: int,
        max_depth: int,
        attr_incl: float,
):
    print(f"Starting training on {len(train)} data samples")
    print(f"= Feedback {'enabled' if feedback else 'disabled'}")
    print(f"= Forest size: {trees}")
    print(f"= Sample size: {samples}")
    print(f"= Max depth: {max_depth}")
    print(f"= Attribute inclusion: {attr_incl}")

    tic = time.perf_counter()
    forest = RandomForest()
    forest.fit(feedback, train, trees, samples, max_depth, attr_incl, False)
    # with open('json_data.json') as file:
    #     forest = RandomForest.from_dict(json.load(file))
    tac = time.perf_counter()
    print(f"Finished training in: {round(tac - tic, 1)}s")

    print()
    print(f"Starting testing on {len(test)} data samples")

    tic = time.perf_counter()
    p = forest.perform(test, False)
    tac = time.perf_counter()
    # with open('json_data.json', 'w') as outfile:
    #     outfile.write(forest.to_json())

    print(f"Finished testing in: {round(tac - tic, 1)}s")
    print(f"Avg. error: {round(p, 1)}")

    p2 = forest.predict(
        np.array([0, 248973, 0.605, 0.882, 9, -3.028, 0, 0.029, 0.0000313, 0.614, 0.135, 0.418, 140.026, 4]))

    print(f"Predicted: {p2}")


def main():
    # index 0 is the value, the rest are the attributes
    a = load_data_from_csv("data/tracks_processed_fix.csv")

    # remove entries with less than 1950 in the first column
    a = a[a[:, 0] >= 1950]

    # remove outliers based on standard deviation in all columns
    # for i in range(1, len(a[0])):
    #     mean = np.mean(a[:, i])
    #     std = np.std(a[:, i])
    #     a = a[np.abs(a[:, i] - mean) < 3 * std]

    # sort by year
    a = a[a[:, 0].argsort()]

    # group by year
    a = np.split(a, np.where(np.diff(a[:, 0]))[0] + 1)

    # ensure that each year has the same number of entries by oversampling
    for i in range(len(a)):
        b = np.random.choice(a[i].shape[0], 2500, replace=True)
        a[i] = a[i][b]

    # flatten
    a = np.concatenate(a)

    # shuffle the data
    np.random.shuffle(a)

    # split the data into training and testing
    dataset_len = len(a)
    training_size = 0.6
    testing_size = 0.07
    training = a[: int(dataset_len * training_size)]
    testing = a[int(dataset_len * training_size): int(dataset_len * (testing_size + training_size))]

    # Model testing
    check(training, testing, True, 50, 5000, 10, 0.34)
    print("\n=======================\n")


if __name__ == "__main__":
    main()
