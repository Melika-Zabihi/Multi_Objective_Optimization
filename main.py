import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from problem import OptimisationProblem
from sampling import CustomSamplingMax
from sklearn import datasets
from mutation import CustomMutation


def main():
    dataset = datasets.load_iris()
    # dataset = datasets.load_digits()
    # dataset = datasets.load_wine()
    # dataset = datasets.load_breast_cancer()

    pop_size = 50

    algorithm = NSGA2(pop_size=pop_size, sampling=CustomSamplingMax())
    problem = OptimisationProblem(dataset=dataset)
    res = minimize(problem, algorithm, ('n_gen', 10), seed=1, verbose=False)
    print(res.F)

    algorithm2 = NSGA2(pop_size=pop_size, sampling=CustomSamplingMax(), mutation=CustomMutation())
    problem = OptimisationProblem(dataset=dataset)
    res2 = minimize(problem, algorithm2, ('n_gen', 10), seed=1, verbose=False)
    print(res2.F)

    plot = Scatter()
    plot.add(res.F, color="red")
    plot.add(res2.F, color="blue")
    plot.show()


if __name__ == "__main__":
    main()
