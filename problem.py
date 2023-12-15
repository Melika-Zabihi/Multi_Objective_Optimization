from pymoo.core.problem import Problem
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from knn import KNN

# dataset = datasets.load_iris()
# dataset = datasets.load_digits()
dataset = datasets.load_wine()
# dataset = datasets.load_breast_cancer()

X = dataset.data
y = dataset.target
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)


class OptimisationProblem(Problem):
    def __init__(self, **kwargs):
        super().__init__(n_var=X_train1.shape[1],
                         n_obj=2,
                         n_constr=0,
                         xl=np.zeros(X_train1.shape[1]),
                         xu=np.ones(X_train1.shape[1]))
        self.counter = 0

    def _evaluate(self, dataset_weights_list, out, *args, **kwargs):

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        accuracies = []
        zero_weights_count = []

        for weights in dataset_weights_list:
            weighted_X = X * weights
            X_train, X_test, y_train, y_test = train_test_split(weighted_X, y, test_size=0.2, random_state=42)

            knn = KNN(k=7)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)*100
            zero_weights = np.sum(weights == 0)

            accuracies.append(-accuracy)
            zero_weights_count.append(-zero_weights)

        print(self.counter)
        self.counter += 1
        print(accuracies)
        print(zero_weights_count)

        out["F"] = np.column_stack([accuracies, zero_weights_count])
