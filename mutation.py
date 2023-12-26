from pymoo.core.mutation import Mutation
from pymoo.core.variable import get, Real
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
import numpy as np


def mut_pm(X, xl, xu, eta, prob, at_least_once):
    n, n_var = X.shape
    assert len(eta) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = mut_binomial(n, n_var, prob, at_least_once=at_least_once)
    mut[:, xl == xu] = False

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]

    X = X[mut]
    eta = np.tile(eta[:, None], (1, n_var))[mut]

    delta1 = (X - _xl) / (_xu - _xl)
    delta2 = (_xu - X) / (_xu - _xl)

    mut_pow = 1.0 / (eta + 1.0)

    rand = np.random.random(X.shape)
    # print("****" + str(X.shape) + "   "+ str(rand))

    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]

    # mutated values
    _Y = X + deltaq * (_xu - _xl)

    # back in bounds if necessary (floating point issues)
    _Y[_Y < _xl] = _xl[_Y < _xl]
    _Y[_Y > _xu] = _xu[_Y > _xu]

    # set the values for output
    Xp[mut] = _Y

    # in case out of bounds repair (very unlikely)
    Xp = set_to_bounds_if_outside(Xp, xl, xu)

    return Xp


def mut_pm2(X, xl, xu, eta, prob, at_least_once):
    n, n_var = X.shape
    assert len(eta) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = mut_binomial(n, n_var, prob, at_least_once=at_least_once)
    mut[:, xl == xu] = False

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]

    X = X[mut]
    eta = np.tile(eta[:, None], (1, n_var))[mut]

    delta1 = (X - _xl) / (_xu - _xl)
    delta2 = (_xu - X) / (_xu - _xl)

    mut_pow = 1.0 / (eta + 1.0)

    rand2 = np.random.exponential(scale=0.5 ,size=X.shape)
    # print("****" + str(X.shape) + "   "+ str(rand2))

    rand = (rand2 - np.min(rand2)) / (np.max(rand2) - np.min(rand2))

    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]

    # mutated values
    _Y = X + deltaq * (_xu - _xl)

    # back in bounds if necessary (floating point issues)
    _Y[_Y < _xl] = _xl[_Y < _xl]
    _Y[_Y > _xu] = _xu[_Y > _xu]

    # set the values for output
    Xp[mut] = _Y

    # in case out of bounds repair (very unlikely)
    Xp = set_to_bounds_if_outside(Xp, xl, xu)

    return Xp


def mut_pm3(X, xl, xu, eta, prob, at_least_once):
    n, n_var = X.shape
    assert len(eta) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = mut_binomial(n, n_var, prob, at_least_once=at_least_once)
    mut[:, xl == xu] = False

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]

    X = X[mut]
    eta = np.tile(eta[:, None], (1, n_var))[mut]

    delta1 = (X - _xl) / (_xu - _xl)
    delta2 = (_xu - X) / (_xu - _xl)

    mut_pow = 1.0 / (eta + 1.0)

    rand = 0
    # print("****" + str(X.shape) + "   "+ str(rand))

    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]

    # mutated values
    _Y = X + deltaq * (_xu - _xl)

    # back in bounds if necessary (floating point issues)
    _Y[_Y < _xl] = _xl[_Y < _xl]
    _Y[_Y > _xu] = _xu[_Y > _xu]

    # set the values for output
    Xp[mut] = _Y

    # in case out of bounds repair (very unlikely)
    Xp = set_to_bounds_if_outside(Xp, xl, xu)

    return Xp


def get_random_floats(X):
    random_floats = np.random.exponential(scale=1 ,size=X.shape)
    min_val = min(random_floats)
    max_val = max(random_floats)
    random_floats = [(x - min_val) / (max_val - min_val) for x in random_floats]
    return random_floats


def get_weighted_random(shape=(1,)):
    choices = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    weights = [0.4, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    rand = np.random.choice(choices, p=weights, size=shape).reshape(shape)
    return rand


class CustomMutation(Mutation):

    def __init__(self, prob=0.9, eta=20, at_least_once=False, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, 100.0))

    def _do(self, problem, X, params=None, **kwargs):

        X = X.astype(float)

        eta = get(self.eta, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))
        # print(X)
        # print("+++")
        # Xp = mut_pm(X, problem.xl, problem.xu, eta, prob_var, at_least_once=self.at_least_once)
        # print(Xp)
        # print("___________")
        Xp2 = mut_pm2(X, problem.xl, problem.xu, eta, prob_var, at_least_once=self.at_least_once)
        # print(X)
        # print("+++")
        # print(Xp2)
        # print("___________")
        # Xp3 = mut_pm3(X, problem.xl, problem.xu, eta, prob_var, at_least_once=self.at_least_once)
        # print(X)
        # print("+++")
        # print(Xp3)
        return Xp2

