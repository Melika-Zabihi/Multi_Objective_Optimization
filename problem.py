import numpy
from pymoo.core.problem import Problem
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from knn import KNN


class OptimisationProblem(Problem):
    def __init__(self, dataset=None, X=None, y=None, random_state=0, **kwargs):
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
        self.random_state = random_state

    def _evaluate(self, dataset_weights_list, out, *args, **kwargs):

        accuracies = []
        weights_count = []
        for weights in dataset_weights_list:
            weighted_X = self.X * weights
            X_train, X_test, y_train, y_test = train_test_split(weighted_X,
                                                                self.y,
                                                                test_size=0.3,
                                                                random_state=self.random_state)
            knn = KNN(k=7)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            reverse_accuracy = (1-accuracy_score(y_test, y_pred)) * 100
            zero_weights = np.sum(weights == 0)
            remaining_weights = len(weights) - zero_weights

            accuracies.append(reverse_accuracy)
            weights_count.append(remaining_weights)

        print(self.counter)
        self.counter += 1
        print(accuracies)
        print(weights_count)

        out["F"] = np.column_stack([accuracies, weights_count])
