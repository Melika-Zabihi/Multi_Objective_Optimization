from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from problem import OptimisationProblem
from sampling import CustomSampling
from mutation import CustomMutation
import numpy as np
import enum

pop_size = 50
n_gen = 10


class Method(enum.Enum):
    normal = 1
    enhanced_mutation = 2
    selection = 3


def get_result(dataset=None, X=None, y=None, method=Method.normal, random_state=0):
    if dataset is not None:
        problem = OptimisationProblem(dataset=dataset, random_state=random_state)
    else:
        problem = OptimisationProblem(X=X, y=y, random_state=random_state)

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


def normalize_data(data):
    # Normalize the first and second elements of data separately
    data_norm = np.empty_like(data)
    for i in range(data.shape[1]):
        min_val = np.min(data[:, i])
        max_val = np.max(data[:, i])
        data_norm[:, i] = (data[:, i] - min_val) / (max_val - min_val)
    return data_norm


class Result:
    def __init__(self, res1, res2, hv1, hv2):
        self.res1 = res1
        self.res2 = res2
        self.hv1 = hv1
        self.hv2 = hv2

    def get_max_acc_res1(self):
        max = 0
        for res in self.res1:
            if res[0] > max:
                max = res[0]
        return max

    def get_max_acc_res2(self):
        max = 0
        for res in self.res2:
            if res[0] > max:
                max = res[0]
        return max