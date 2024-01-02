from utility import *
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV


def main():
    run_count = 5
    results = []
    for i in range(run_count):
        random_state = i
        res, res2 = get_sklearn_dataset_res(random_state)
        res_u = np.unique(res.F, axis=0)
        res2_u = np.unique(res2.F, axis=0)
        print(res_u)
        print(res2_u)
        res_abs = np.abs(res_u)
        res2_abs = np.abs(res2_u)

        fig, ax = plt.subplots()

        scatter1 = ax.scatter(res_abs[:, 0], res_abs[:, 1], color="red", label='Selection')
        scatter2 = ax.scatter(res2_abs[:, 0], res2_abs[:, 1], color="blue", label='Weighting')

        ax.invert_xaxis()
        ax.invert_yaxis()

        plt.xlabel('Accuracy')
        plt.ylabel('Removed Features')
        plt.legend(handles=[scatter1, scatter2])
        plt.show()

        hv1, hv2 = calculate_hv(res_abs,res2_abs)

        print("HV Selection", hv1)
        print("HV Weighting", hv2)
        results.append(Result(res_abs, res2_abs, hv1=hv1, hv2=hv2))

    for i in results:
        print("selection:")
        print(i.get_max_acc_res1())
        print(i.hv1)
        print("weighting")
        print(i.get_max_acc_res2())
        print(i.hv2)
        print("+++++++++++")


def get_uci_dataset_res(random_state):
    data = pd.read_csv("dataset\\arrhythmia\\arrhythmia.data", na_values="?")
    data = data.fillna(value=0)

    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    res = get_result(X=X,
                     y=y,
                     method=Method.selection,
                     random_state=random_state)

    res2 = get_result(X=X,
                      y=y,
                      method=Method.enhanced_mutation,
                      random_state=random_state)
    return res, res2


def calculate_hv(res, res2):
    res_norm = 1-normalize_data(res)
    res2_norm = 1-normalize_data(res2)
    ref_point = np.array([1.1, 1.1])

    ind = HV(ref_point=ref_point)
    hv1 = ind(res_norm)
    hv2 = ind(res2_norm)
    return hv1, hv2

def get_sklearn_dataset_res(random_state):
    # dataset = datasets.load_iris()
    # dataset = datasets.load_digits()
    dataset = datasets.load_wine()
    # dataset = datasets.load_breast_cancer()

    res = get_result(dataset=dataset,
                     method=Method.selection,
                     random_state=random_state)

    res2 = get_result(dataset=dataset,
                      method=Method.enhanced_mutation,
                      random_state=random_state)

    return res, res2


if __name__ == "__main__":
    main()
