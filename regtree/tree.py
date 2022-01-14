from __future__ import annotations
from abc import ABC, abstractmethod, abstractstaticmethod 
from regtree.utils import neighbor_avg
import typing
import numpy as np
from math import floor

class RandomForest():
    data: np.ndarray
    trees: typing.List[RegressionTree]

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def generate_trees(self, tree_count, sample_size) -> None:
        self.trees = []
        for _ in range(tree_count):
            data_points = self.data[np.random.choice(self.data.shape[0], sample_size, replace=False)]
            tree = RegressionTree.fit(data_points)
            self.trees.append(tree)

    def predict(self, attributes:np.ndarray) -> np.float64:
        a = []
        for t in self.trees:
            a.append(t.predict(attributes))
        return np.array(a).mean()

    def perform(self, data:np.ndarray) -> np.float64:
        predicted = []
        for d in data:
            predicted.append(self.predict(d[1:]))

        print(predicted)
        return (np.abs((data[:,0] - predicted) / data[:,0])).mean()


class RegressionTree():
    root: RegressionNode

    def __init__(self, root: RegressionNode) -> None:
        self.root = root
        pass

    def __str__(self) -> str:
        return f"RegressionTree[root={self.root}]"

    def predict(self, attributes: np.ndarray) -> np.float64:
        return self.root.predict(attributes)

    def depth(self) -> int:
        return self.root.depth()

    def pretty(self) -> str:
        return self.root.pretty((2 ** (self.depth() - 1) - 1) * 6)

    def to_dict(self):
        return self.root.to_dict()

    @staticmethod
    def from_dict(d):
        root = RegressionElement.from_dict(d)
        if root is not RegressionNode:
            raise ValueError("Root has to be a node")

        return RegressionTree(root)

    @staticmethod
    def fit(array: np.ndarray, attr_incl:float=1):
        root = RegressionNode.make_tree(array, attr_incl)
        if root is None:
            raise ValueError("Empty tree")

        return RegressionTree(root)


class RegressionElement(ABC):
    @abstractmethod
    def predict(self, attributes: np.ndarray) -> np.float64:
        pass

    @abstractmethod
    def pretty(self, o:int = 0) -> str:
        pass

    @abstractmethod
    def child_count(self) -> int:
        pass

    @abstractmethod
    def depth(self) -> int:
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @staticmethod
    def from_dict(d) -> RegressionElement:
        if d['type'] == 'node':
            node = RegressionNode(d['attr'], d['value'])
            node.children = (
                RegressionNode.from_dict(d['children'][0]), 
                RegressionNode.from_dict(d['children'][0])
            )
            return node
        elif d['type'] == 'leaf':
            return RegressionLeaf(d['value'])
        else:
            raise ValueError("Unknown regression tree type")

class RegressionLeaf(RegressionElement):
    value: np.float64

    def __init__(self, value: np.float64) -> None:
        self.value = value


    def __str__(self) -> str:
        return f"RegressionLeaf[v={self.value}]"


    def __format__(self, format_spec: str) -> str:
        return self.__str__()


    def __repr__(self) -> str:
        return self.__str__()


    def predict(self, attributes: np.ndarray) -> np.float64:
        return self.value


    def pretty(self, o:int=0) -> str:
        v = str(self.value)[:7]
        if len(v) & 1 == 1:
            if "." in v:
                v += "0"
            else:
                v += "."
        i = round((8 - len(v)) / 2)

        s = ""
        s += " " * o + "|== Leaf ==|"               + " " * o + "\n" + \
             " " * o + "| " + " " * i + v + " " * i +" |" + " " * o + "\n" + \
             " " * o + "|==========|"               + " " * o + "\n" + \
             " " * (2 * o + 12) + "\n"

        return s


    def child_count(self) -> int:
        return 0


    def depth(self) -> int:
        return 1

    def to_dict(self):
        return {
            'type': 'leaf',
            'value': self.value
        }


class RegressionNode(RegressionElement):
    attr: int
    value: np.float64
    children: typing.Tuple[RegressionElement, RegressionElement]

    def __init__(self, attr: int, value: np.float64) -> None:
        if attr < 1:
            raise ValueError("Attributes start from index 1")

        self.attr = attr
        self.value = value


    def __str__(self) -> str:
        return f"RegressionNode[a={self.attr}, v={self.value}, c={self.children}]"


    def __format__(self, format_spec: str) -> str:
        return self.__str__()


    def __repr__(self) -> str:
        return self.__str__()


    def split(self, array: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        if array.shape[1] <= self.attr:
            raise IndexError("Attribute index out of bounds")

        return (
            array[np.argwhere(array[:, self.attr] <= self.value).flatten()],
            array[np.argwhere(array[:, self.attr] > self.value).flatten()]
        )

    def mse(self, array: np.ndarray) -> np.float64:
        split = self.split(array)

        split1 = split[0][:, 0]
        split2 = split[1][:, 0]

        sigma1 = split1.mean()
        sigma2 = split2.mean()

        return np.square(np.concatenate([split1 - sigma1, split2 - sigma2])).mean()


    def create_nodes(self, array: np.ndarray, attr_incl:float=1) -> None:
        split = self.split(array)

        node_l = RegressionNode.best_split(split[0], attr_incl)
        node_r = RegressionNode.best_split(split[1], attr_incl)

        if node_l is None:
            node_l = RegressionLeaf(split[0][0, 0])
        else:
            node_l.create_nodes(split[0], attr_incl)

        if node_r is None:
            node_r = RegressionLeaf(split[1][0, 0])
        else:
            node_r.create_nodes(split[1], attr_incl)

        self.children = (node_l, node_r)


    def predict(self, attributes: np.ndarray) -> np.float64:
        if attributes[self.attr - 1] <= self.value:
            return self.children[0].predict(attributes)
        return self.children[1].predict(attributes)


    def pretty(self, o:int = 0) -> str:
        v1 = f"a={self.attr}"
        if len(v1) & 1 == 1:
            if "." in v1:
                v1 += "0"
            else:
                v1 += "."
        v2 = f"v={self.value}"
        if len(v2) & 1 == 1:
            if "." in v2:
                v2 += "0"
            else:
                v2 += "."
        i1 = round((8 - len(v1)) / 2)
        i2 = round((8 - len(v2)) / 2)

        s = ""
        s += " " * o + "|== Node ==|" +               " " * o + "\n" + \
             " " * o + "| " + " " * i1 + v1 + " " * i1 +" |" + " " * o + "\n" + \
             " " * o + "| " + " " * i2 + v2 + " " * i2 +" |" + " " * o + "\n" + \
             " " * o + "|==========|" +               " " * o + "\n"

        c1 = self.children[0].pretty(floor((o - 6) / 2))
        c2 = self.children[1].pretty(floor((o - 6) / 2))

        cs1 = c1.split("\n")
        cs2 = c2.split("\n")

        m = max(len(cs1), len(cs2))
        cs1 += [""] * (m - len(cs1))
        cs2 += [""] * (m - len(cs2))


        s += "\n".join(["".join(x) for x in list(zip(cs1, cs2))])

        return s

    def child_count(self) -> int:
        return 2 + self.children[0].child_count() + self.children[1].child_count()

    def depth(self) -> int:
        return 1 + max(self.children[0].depth(), self.children[1].depth())

    def to_dict(self):
        return {
            'type': 'node',
            'attr': self.attr,
            'value': self.value,
            'children': [
                self.children[0].to_dict(),
                self.children[1].to_dict()
            ]
        }


    @staticmethod
    def best_split(array: np.ndarray, attr_incl:float=1) -> RegressionNode | None:
        best = (np.Infinity, None)

        attrc = max(1, round((array.shape[1] - 1) * attr_incl))
        attrs = np.random.choice(array.shape[1] - 1, attrc) + 1

        for attr in attrs:
            for split in neighbor_avg(array[:, attr]):
                node = RegressionNode(attr, split)
                mse = node.mse(array)
                if mse < best[0]:
                    best = (mse, node)

        return best[1]

    @staticmethod
    def make_tree(array: np.ndarray, attr_incl:float=1) -> RegressionNode | None:
        root = RegressionNode.best_split(array, attr_incl)
        if root is None:
            return None

        root.create_nodes(array, attr_incl)

        return root

