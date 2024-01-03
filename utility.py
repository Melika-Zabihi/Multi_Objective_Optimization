from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from problem import OptimisationProblem
from sampling import CustomSampling
from mutation import CustomMutation
from pymoo.indicators.hv import HV
import numpy as np
import enum

pop_size = 30
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

    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=1, verbose=False)
    return res


def calculate_hv(res, res2):
    res_norm = normalize_data(res)
    res2_norm = normalize_data(res2)
    ref_point = np.array([1.1, 1.1])
    ind = HV(ref_point=ref_point)
    hv1 = ind(res_norm)
    hv2 = ind(res2_norm)
    return hv1, hv2


def normalize_data(res):
    data = res.F
    data_norm = np.empty_like(data)
    data_norm[:, 0] = data[:, 0] / 100
    data_norm[:, 1] = data[:, 1] / res.problem.n_var
    return data_norm


class Result:
    def __init__(self, res1, res2, hv1, hv2):
        self.res1 = res1
        self.res2 = res2
        self.hv1 = hv1
        self.hv2 = hv2

    def get_min_rev_acc_res1(self):
        min_acc = 100
        features = 0
        for res in self.res1:
            if res[0] <= min_acc:
                min_acc = res[0]
                features = res[1]
        return min_acc, features

    def get_min_rev_acc_res2(self):
        min_acc = 100
        features = 0
        for res in self.res2:
            if res[0] <= min_acc:
                min_acc = res[0]
                features = res[1]
        return min_acc, features
