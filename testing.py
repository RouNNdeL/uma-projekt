#!/usr/bin/env python3

from regtree.tree import RandomForest
import numpy as np
from utils import load_data_from_csv
import time
import matplotlib.pyplot as plt


def train(
        train,
        test,
        feedback: bool,
        trees: int,
        samples: int,
        max_depth: int,
        attr_incl: float,
):
    forest = RandomForest()
    forest.fit(feedback, train, trees, samples, max_depth, attr_incl)

    return forest


def test_time(
        training,
        testing,
        max=100,
        interval=10,
        start=1,
):
    times = []
    r = range(start, max, interval)
    for x in r:
        start = time.time()
        forest = train(
            training,
            testing,
            feedback=True,
            trees=25,
            samples=500,
            max_depth=10,
            attr_incl=x/r[-1],
        )
        end = time.time()
        times.append(end - start)
        # print progress
        print(f"\r{x} trees: {end - start} seconds")

    # plot the training time
    plt.plot([x/r[-1] for x in r], times)
    plt.title(f"Training time based on attribute inclusion")
    plt.xlabel("Attribute inclusion")
    plt.ylabel("Training time (s)")
    plt.savefig("time_attr.png")
    plt.show()

def test_accuracy(
        training,
        testing,
        max=100,
        interval=10,
        start=1,
):
    accuracies = []
    r = range(start, max, interval)
    for x in r:
        forest = train(
            training,
            testing,
            feedback=True,
            trees=25,
            samples=1000,
            max_depth=10,
            attr_incl=0.34,
        )
        accuracy = forest.perform(testing)
        accuracies.append(accuracy)
        # print progress
        print(f"\r{x} samples: {accuracy}")

    # plot the accuracy
    plt.plot([x/r[-1] for x in r], accuracies)
    plt.title(f"Accuracy based on attribute inclusion")
    plt.xlabel("Attribute inclusion")
    plt.ylabel("Accuracy")

    plt.savefig("attr.png")
    plt.show()

def main():
    # index 0 is the value, the rest are the attributes
    a = load_data_from_csv("data/winequality-red.csv", delimiter=";")

    # shuffle the data
    np.random.shuffle(a)

    # split the data into training and testing
    training = a[: int(len(a) * 0.66)]
    testing = a[int(len(a) * 0.66):]

    # Model testing

    # test the model training time with different number of trees
    # test_time(training, testing, 250, 25)

    # test the model accuracy with different number of trees
    test_time(training, testing, 10, 1, 1)

if __name__ == "__main__":
    main()
