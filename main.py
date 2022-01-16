#!/usr/bin/env python3

from regtree.utils import NpEncoder
from regtree.tree import RandomForest, RegressionTree
import numpy as np
from utils import load_data_from_csv
import json
import time

def main():
    # index 0 is the value, the rest are the attributes
    a = load_data_from_csv("data/cal_housing.data")

    # shuffle the data
    np.random.shuffle(a)

    # split the data into training and testing
    training = a[:int(len(a)*0.66)]
    testing = a[int(len(a)*0.66):]

    # train the model
    tic = time.perf_counter()
    forest = RandomForest(training)
    forest.generate_trees_feedback(10, 5000)
    tac = time.perf_counter()
    print(f"Time to train: {tac-tic}")

    # test the model
    p = forest.perform(testing)

    print(p)



if __name__ == "__main__":
    main()
