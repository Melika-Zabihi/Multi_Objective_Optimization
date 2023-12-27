import random
import numpy as np
from pymoo.core.sampling import Sampling

pop_size = 50


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
