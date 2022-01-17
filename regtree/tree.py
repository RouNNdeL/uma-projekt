from __future__ import annotations

import json
from abc import ABC, abstractmethod
from sys import stdout
from typing import Dict, Tuple, List, Any

import numpy as np

from regtree.utils import NpEncoder, neighbor_avg


class RandomForest:
    __trees: List[RegressionTree]

    def fit(
            self,
            use_feedback: bool,
            data: np.ndarray,
            tree_count: int,
            sample_size: int,
            max_depth: int = 0,
            attr_incl: float = 1.0,
            percentage=True,
    ) -> None:
        if use_feedback:
            self.__fit_feedback(data, tree_count, sample_size, max_depth, attr_incl, percentage)
        else:
            self.__fit_normal(data, tree_count, sample_size, max_depth, attr_incl)

    def predict(self, attributes: np.ndarray) -> np.float64:
        a = []
        for t in self.__trees:
            a.append(t.predict(attributes))
        return np.array(a).mean()

    def perform(self, data: np.ndarray, percentage=True) -> np.float64:
        predicted = []
        for d in data:
            predicted.append(self.predict(d))
        if percentage:
            return (np.abs((data[:, 0] - predicted) / data[:, 0])).mean()
        return (np.abs(data[:, 0] - predicted)).mean()

    def to_dict(self) -> Dict[str, Any]:
        return {"trees": [t.to_dict() for t in self.__trees]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), cls=NpEncoder)

    def __best_performers(self, data: np.ndarray, n: int, percentage=True) -> Tuple[np.float64, np.ndarray]:
        predictions = np.apply_along_axis(self.predict, 1, data)
        if percentage:
            performance = np.abs(data[:, 0] - predictions) / data[:, 0]
        else:
            performance = np.abs(data[:, 0] - predictions)
        # choose the worst performing trees randomly with weighting
        # based on the performance
        worst = np.random.choice(data.shape[0], n, False, performance / performance.sum())

        return performance.mean(), data[worst]

    def __fit_normal(
            self,
            data: np.ndarray,
            tree_count: int,
            sample_size: int,
            max_depth: int,
            attr_incl: float,
    ) -> None:
        self.__trees = []
        for i in range(tree_count):
            data_points = data[
                np.random.choice(data.shape[0], sample_size, replace=False)
            ]
            tree = RegressionTree.fit(data_points, max_depth, attr_incl)
            self.__trees.append(tree)
            stdout.write(f"\r- tree: {i + 1}")
            stdout.flush()
        stdout.write(f"\r")
        stdout.flush()

    def __fit_feedback(
            self,
            data: np.ndarray,
            tree_count: int,
            sample_size: int,
            max_depth: int,
            attr_incl: float,
            percentage=True,
    ) -> None:
        self.__trees = []
        data_points = data[np.random.choice(data.shape[0], sample_size, replace=False)]
        for i in range(tree_count):
            tree = RegressionTree.fit(data_points, max_depth, attr_incl)
            self.__trees.append(tree)
            performance = self.__best_performers(data, sample_size, percentage)
            data_points = performance[1]
            stdout.write(
                f"\r- tree: {i + 1} perf: {round(performance[0].mean(), 2)}  "
            )
            stdout.flush()
        stdout.write(f"\r")
        stdout.flush()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> RandomForest:
        trees = []
        for t in d["trees"]:
            trees.append(RegressionTree.from_dict(t))

        forest = RandomForest()
        forest.__trees = trees
        return forest


class RegressionTree:
    __root: RegressionNode

    def __init__(self, root: RegressionNode) -> None:
        self.__root = root
        pass

    def __str__(self) -> str:
        return f"RegressionTree[root={self.__root}]"

    def predict(self, attributes: np.ndarray) -> np.float64:
        return self.__root.predict(attributes)

    def to_dict(self):
        return self.__root.to_dict()

    @staticmethod
    def from_dict(d):
        root = RegressionElement.from_dict(d)
        if not isinstance(root, RegressionNode):
            raise ValueError("Root has to be a node")

        return RegressionTree(root)

    @staticmethod
    def fit(array: np.ndarray, max_depth: int, attr_incl: float = 1):
        root = RegressionNode.make_tree(array, max_depth, attr_incl)
        if root is None:
            raise ValueError("Empty tree")

        return RegressionTree(root)


class RegressionElement(ABC):
    @abstractmethod
    def predict(self, attributes: np.ndarray) -> np.float64:
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @staticmethod
    def from_dict(d, depth: int = 0) -> RegressionElement:
        if d["type"] == "node":
            node = RegressionNode(d["attr"], d["value"], depth, 0, (
                RegressionNode.from_dict(d["childl"], depth + 1),
                RegressionNode.from_dict(d["childr"], depth + 1),
            ))
            return node
        elif d["type"] == "leaf":
            return RegressionLeaf(d["value"])
        else:
            raise ValueError("Unknown regression tree type")


class RegressionLeaf(RegressionElement):
    __value: np.float64

    def __init__(self, value: np.float64) -> None:
        self.__value = value

    def __str__(self) -> str:
        return f"RegressionLeaf[v={self.__value}]"

    def __format__(self, format_spec: str) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def predict(self, attributes: np.ndarray) -> np.float64:
        return self.__value

    def to_dict(self):
        return {"type": "leaf", "value": self.__value}


class RegressionNode(RegressionElement):
    __attr: int
    __value: np.float64
    __children: Tuple[RegressionElement, RegressionElement]
    __depth: int
    __max_depth: int

    def __init__(
            self, attr: int, value: np.float64, depth: int, max_depth: int,
            children: Tuple[RegressionElement, RegressionElement] = None
    ) -> None:
        if attr < 1:
            raise ValueError("Attributes start from index 1")

        self.__attr = attr
        self.__value = value
        self.__depth = depth
        self.__max_depth = max_depth
        self.__children = children

    def __str__(self) -> str:
        return f"RegressionNode[a={self.__attr}, v={self.__value}, c={self.__children}]"

    def __format__(self, format_spec: str) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def __split(self, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if array.shape[1] <= self.__attr:
            raise IndexError("Attribute index out of bounds")

        return (
            array[np.argwhere(array[:, self.__attr] <= self.__value).flatten()],
            array[np.argwhere(array[:, self.__attr] > self.__value).flatten()],
        )

    def __mse(self, array: np.ndarray) -> np.float64:
        split = self.__split(array)

        split1 = split[0][:, 0]
        split2 = split[1][:, 0]

        sigma1 = split1.mean()
        sigma2 = split2.mean()

        return np.square(np.concatenate([split1 - sigma1, split2 - sigma2])).mean()

    def create_nodes(self, array: np.ndarray, attr_incl: float) -> None:
        split = self.__split(array)

        node_l = RegressionNode.best_split(
            split[0], self.__depth + 1, self.__max_depth, attr_incl
        )
        node_r = RegressionNode.best_split(
            split[1], self.__depth + 1, self.__max_depth, attr_incl
        )

        if node_l is None:
            node_l = RegressionLeaf(split[0][:, 0].mean())
        else:
            node_l.create_nodes(split[0], attr_incl)

        if node_r is None:
            node_r = RegressionLeaf(split[1][:, 0].mean())
        else:
            node_r.create_nodes(split[1], attr_incl)

        self.__children = (node_l, node_r)

    def predict(self, attributes: np.ndarray) -> np.float64:
        if attributes[self.__attr] <= self.__value:
            return self.__children[0].predict(attributes)
        return self.__children[1].predict(attributes)

    def to_dict(self):
        return {
            "type": "node",
            "attr": self.__attr,
            "value": self.__value,
            "childl": self.__children[0].to_dict(),
            "childr": self.__children[1].to_dict(),
        }

    @staticmethod
    def best_split(
            array: np.ndarray, depth: int, max_depth: int, attr_incl: float
    ) -> RegressionNode | None:
        if 0 < max_depth < depth:
            return None

        best = (np.Infinity, None)

        attrc = max(1, round((array.shape[1] - 1) * attr_incl))
        attrs = np.random.choice(array.shape[1] - 1, attrc) + 1

        for attr in attrs:
            for split in neighbor_avg(array[:, attr]):
                node = RegressionNode(attr, split, depth, max_depth)
                mse = node.__mse(array)
                if mse < best[0]:
                    best = (mse, node)

        return best[1]

    @staticmethod
    def make_tree(
            array: np.ndarray, max_depth: int, attr_incl: float
    ) -> RegressionNode | None:
        root = RegressionNode.best_split(array, 0, max_depth, attr_incl)
        if root is None:
            return None

        root.create_nodes(array, attr_incl)

        return root
