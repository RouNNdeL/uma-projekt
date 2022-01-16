from __future__ import annotations

from typing import Dict, Tuple, List, Any
from abc import ABC, abstractmethod

import numpy as np

from regtree.utils import neighbor_avg


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
    ) -> None:
        if use_feedback:
            self.__fit_feedback(data, tree_count, sample_size, max_depth, attr_incl)
        else:
            self.__fit_normal(data, tree_count, sample_size, max_depth, attr_incl)

    def predict(self, attributes: np.ndarray) -> np.float64:
        a = []
        for t in self.__trees:
            a.append(t.predict(attributes))
        return np.array(a).mean()

    def perform(self, data: np.ndarray) -> np.float64:
        predicted = []
        for d in data:
            predicted.append(self.predict(d[1:]))
        return (np.abs((data[:, 0] - predicted) / data[:, 0])).mean()

    def to_dict(self) -> Dict[str, Any]:
        return {"trees": [t.to_dict() for t in self.__trees]}

    def __worst_bootstrap(self, data: np.ndarray) -> np.ndarray:
        bootstraps: List[Tuple[np.float64, np.ndarray]] = []
        for d in data:
            predicted = self.predict(d[1:])
            bootstraps.append((abs((d[0] - predicted) / d[0]), d))
        # sort bootstraps by error descending
        bootstraps.sort(key=lambda x: x[0], reverse=True)
        return np.array([b[1] for b in bootstraps])

    def __fit_normal(
        self,
        data: np.ndarray,
        tree_count: int,
        sample_size: int,
        max_depth: int,
        attr_incl: float,
    ) -> None:
        self.__trees = []
        for _ in range(tree_count):
            data_points = data[
                np.random.choice(data.shape[0], sample_size, replace=False)
            ]
            tree = RegressionTree.fit(data_points, max_depth, attr_incl)
            self.__trees.append(tree)

    def __fit_feedback(
        self,
        data: np.ndarray,
        tree_count: int,
        sample_size: int,
        max_depth: int,
        attr_incl: float,
    ) -> None:
        self.__trees = []
        data_points = data[np.random.choice(data.shape[0], sample_size, replace=False)]
        for _ in range(tree_count):
            tree = RegressionTree.fit(data_points, max_depth, attr_incl)
            self.__trees.append(tree)
            data_points = self.__worst_bootstrap(data)[:sample_size]

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
        if root is not RegressionNode:
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
            node = RegressionNode(d["attr"], d["value"], depth, 0)
            node.__children = (
                RegressionNode.from_dict(d["children"][0], depth + 1),
                RegressionNode.from_dict(d["children"][1], depth + 1),
            )
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
        self, attr: int, value: np.float64, depth: int, max_depth: int
    ) -> None:
        if attr < 1:
            raise ValueError("Attributes start from index 1")

        self.__attr = attr
        self.__value = value
        self.__depth = depth
        self.__max_depth = max_depth

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
        if attributes[self.__attr - 1] <= self.__value:
            return self.__children[0].predict(attributes)
        return self.__children[1].predict(attributes)

    def to_dict(self):
        return {
            "type": "node",
            "attr": self.__attr,
            "value": self.__value,
            "children": [self.__children[0].to_dict(), self.__children[1].to_dict()],
        }

    @staticmethod
    def best_split(
        array: np.ndarray, depth: int, max_depth: int, attr_incl: float
    ) -> RegressionNode | None:
        if max_depth > 0 and depth > max_depth:
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
