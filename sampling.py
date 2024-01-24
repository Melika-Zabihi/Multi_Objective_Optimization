import random
import numpy as np
from pymoo.core.sampling import Sampling


class CustomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        n_features = problem.n_var
        rand_data = np.random.random((n_samples, n_features))
        for i in range(n_samples):
            zero_count = random.randint(0, n_features)
            indices = np.random.choice(np.arange(n_features), replace=False, size=zero_count)
            for index in indices:
                rand_data[i][index] = 0
        return rand_data


class CustomSelectionSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        n_features = problem.n_var
        data = np.ones((n_samples, n_features))
        for i in range(n_samples):
            zero_count = random.randint(0, n_features)
            indices = np.random.choice(np.arange(n_features), replace=False, size=zero_count)
            for index in indices:
                data[i][index] = 0
        return data.astype(bool)