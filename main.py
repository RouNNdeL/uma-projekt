#!/usr/bin/env python3

from regtree.tree import RandomForest
import numpy as np
from utils import load_data_from_csv
import time


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
    forest.fit(feedback, train, trees, samples, max_depth, attr_incl)
    tac = time.perf_counter()
    print(f"Finished training in: {round(tac-tic, 1)}s")

    print()
    print(f"Starting testing on {len(test)} data samples")

    tic = time.perf_counter()
    p = forest.perform(test)
    tac = time.perf_counter()

    print(f"Finished testing in: {round(tac-tic, 1)}s")
    print(f"Avg. error: {round(p * 100, 2)}%")


def main():
    # index 0 is the value, the rest are the attributes
    a = load_data_from_csv("data/cal_housing.data")

    # shuffle the data
    np.random.shuffle(a)

    # split the data into training and testing
    training = a[: int(len(a) * 0.66)]
    testing = a[int(len(a) * 0.66) :]

    # Model testing
    check(training, testing, False, 25, 1000, 3, 0.34)
    print("\n=======================\n")
    check(training, testing, False, 25, 1000, 4, 0.34)
    print("\n=======================\n")
    check(training, testing, False, 25, 1000, 5, 0.34)
    print("\n=======================\n")
    check(training, testing, False, 25, 1000, 6, 0.34)
    print("\n=======================\n")
    check(training, testing, False, 25, 1000, 7, 0.34)
    print("\n=======================\n")
    check(training, testing, False, 25, 1000, 8, 0.34)
    print("\n=======================\n")
    check(training, testing, False, 25, 1000, 9, 0.34)
    print("\n=======================\n")
    check(training, testing, False, 25, 1000, 10, 0.34)
    print("\n=======================\n")
    check(training, testing, False, 25, 1000, 0, 0.34)
    print("\n=======================\n")


if __name__ == "__main__":
    main()
