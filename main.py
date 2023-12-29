from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from problem import OptimisationProblem
from sampling import CustomSampling
from sklearn import datasets
from mutation import CustomMutation
import enum
import pandas as pd

pop_size = 100 #
n_gen = 50


def main():

    res, res2 = uci_datasets()
    print(res.F)
    print(res2.F)

    plot = Scatter()
    plot.add(res.F, color="red")
    plot.add(res2.F, color="blue")
    plot.show()


class Method(enum.Enum):
    normal = 1
    enhanced_mutation = 2
    selection = 3


def uci_datasets():
    data = pd.read_csv("dataset\\arrhythmia\\arrhythmia.data", na_values="?")
    data = data.fillna(value=0)

    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    res, res2 = get_result(X=X, y=y)
    return res, res2


def sklearn_datasets():
    # dataset = datasets.load_iris()
    # dataset = datasets.load_digits()
    # dataset = datasets.load_wine()
    dataset = datasets.load_breast_cancer()

    res = get_result(dataset, Method.selection)
    print(res.F)

    res2 = get_result(dataset, Method.enhanced_mutation)
    print(res2.F)

    return res, res2


def get_result(dataset=None, X=None, y=None, method=Method.normal):

    if dataset is not None:
        problem = OptimisationProblem(dataset=dataset)
    else:
        problem = OptimisationProblem(X=X, y=y)

    if method == Method.enhanced_mutation:
        algorithm = NSGA2(pop_size=pop_size,
                          sampling=CustomSampling(),
                          mutation=CustomMutation())

    elif method == Method.selection:
        algorithm = NSGA2(pop_size=pop_size,
                          sampling=BinaryRandomSampling(),
                          crossover=SinglePointCrossover(),
                          mutation=BitflipMutation())

    elif method == Method.normal:
        algorithm = NSGA2(pop_size=pop_size,
                          sampling=CustomSampling())

    res = minimize(problem, algorithm, ('n_gen', 10), seed=1, verbose=False)

    return res


if __name__ == "__main__":
    main()
