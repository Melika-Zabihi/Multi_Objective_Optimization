import numpy
from pymoo.core.problem import Problem
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from knn import KNN


class OptimisationProblem(Problem):
    def __init__(self, dataset=None, X=None, y=None, **kwargs):
        if dataset is not None:
            self.X = dataset.data
            self.y = dataset.target
        else:
            self.X = X
            self.y = y
        super().__init__(n_var=self.X.shape[1],
                         n_obj=2,
                         n_constr=0,
                         xl=np.zeros(self.X.shape[1]),
                         xu=np.ones(self.X.shape[1]))
        self.counter = 0

    def _evaluate(self, dataset_weights_list, out, *args, **kwargs):

        accuracies = []
        zero_weights_count = []
        for weights in dataset_weights_list:
            weighted_X = self.X * weights
            X_train, X_test, y_train, y_test = train_test_split(weighted_X,
                                                                self.y,
                                                                test_size=0.2,
                                                                random_state=42)
            knn = KNN(k=7)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred) * 100
            zero_weights = np.sum(weights == 0)

            accuracies.append(-accuracy)
            zero_weights_count.append(-zero_weights)

        print(self.counter)
        self.counter += 1
        print(accuracies)
        print(zero_weights_count)

        out["F"] = np.column_stack([accuracies, zero_weights_count])
        # out["H"] = np.column_stack([constraint])
