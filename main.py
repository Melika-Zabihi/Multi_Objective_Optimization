import numpy as np
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

pop_size = 50


def main():
    # dataset = datasets.load_iris()
    # dataset = datasets.load_digits()
    # dataset = datasets.load_wine()
    dataset = datasets.load_breast_cancer()

    res = get_result(dataset, Method.selection)
    print(res.F)

    res2 = get_result(dataset, Method.enhanced_mutation)
    print(res2.F)

    plot = Scatter()
    plot.add(res.F, color="red")
    plot.add(res2.F, color="blue")

    plot.show()


def get_result(dataset, method):
    problem = OptimisationProblem(dataset=dataset)

    if method == Method.normal:
        algorithm = NSGA2(pop_size=pop_size,
                          sampling=CustomSampling())

    elif method == Method.enhanced_mutation:
        algorithm = NSGA2(pop_size=pop_size,
                          sampling=CustomSampling(),
                          mutation=CustomMutation())

    elif method == Method.selection:
        algorithm = NSGA2(pop_size=pop_size,
                          sampling=BinaryRandomSampling(),
                          crossover=SinglePointCrossover(),
                          mutation=BitflipMutation())

    res = minimize(problem, algorithm, ('n_gen', 10), seed=1, verbose=False)
    return res


class Method(enum.Enum):
    normal = 1
    enhanced_mutation = 2
    selection = 3


if __name__ == "__main__":
    main()
