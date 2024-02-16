from utility import *
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from random import randint

def main():
    results = []
    file_name = "results.txt"
    run_count = 4
    for i in range(run_count):
        with open(file_name, 'a') as f:
            random_state = randint(1, 10000)
            print("itr:" + str(i) + " random state:" + str(random_state))
            res, res2 = get_uci_dataset_res(random_state)
            res_u = np.unique(res.F, axis=0)
            res2_u = np.unique(res2.F, axis=0)
            print(res_u)
            print(res2_u)

            hv1, hv2 = calculate_hv(res, res2)
            print("HV Selection", hv1)
            print("HV Weighting", hv2)

            line = "\"" + str(random_state) + "\": {" + " \n"\
                   + "    \"Res1\":" + str(res_u.tolist()) + ',\n' \
                    + "    \"Res2\":" +str(res2_u.tolist()) + ',\n' \
                    + "    \"Hv1\":" + str(hv1) + ',\n' \
                    + "    \"Hv2\":" + str(hv2) + '\n' + " },"

            f.write(line + '\n')

        results.append(Result(res_u, res2_u, hv1=hv1, hv2=hv2))

    extract_output(results)


def get_uci_dataset_res(random_state):
    data = get_11Tumor_dataset()
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    res = get_result(X=X,
                     y=y,
                     method=Method.selection,
                     random_state=random_state)

    res2 = get_result(X=X,
                      y=y,
                      method=Method.normal,
                      random_state=random_state)
    return res, res2

def get_sklearn_dataset_res(random_state):
    dataset = datasets.load_iris()
    # dataset = datasets.load_digits()
    # dataset = datasets.load_wine()
    # dataset = datasets.load_breast_cancer()

    res = get_result(dataset=dataset,
                     method=Method.selection,
                     random_state=random_state)

    res2 = get_result(dataset=dataset,
                      method=Method.normal,
                      random_state=random_state)

    return res, res2


def extract_output(results):
    hv1_sum = 0
    hv2_sum = 0
    max_acc1_sum = 0
    acc_features1_sum = 0
    max_acc2_sum = 0
    acc_features2_sum = 0
    for res in results:
        acc1, features1 = res.get_min_rev_acc_res1()
        max_acc1_sum += acc1
        acc_features1_sum += features1
        hv1_sum += res.hv1
        acc2, features2 = res.get_min_rev_acc_res2()
        max_acc2_sum += acc2
        acc_features2_sum += features2
        hv2_sum += res.hv2

    hv1_avg = hv1_sum / len(results)
    hv2_avg = hv2_sum / len(results)
    max_acc1_avg = max_acc1_sum / len(results)
    max_acc2_avg = max_acc2_sum / len(results)
    features1_avg = acc_features1_sum / len(results)
    features2_avg = acc_features2_sum / len(results)

    print(f"hv1_avg = {hv1_avg:.2f}")
    print(f"hv2_avg = {hv2_avg:.2f}")
    print(f"max_acc1_avg = {max_acc1_avg:.2f}")
    print(f"max_acc2_avg = {max_acc2_avg:.2f}")
    print(f"features1_avg = {features1_avg:.2f}")
    print(f"features2_avg = {features2_avg:.2f}")

    sorted_list1 = sorted(results, key=lambda x: x.hv1)
    sorted_list2 = sorted(results, key=lambda x: x.hv2)

    middle_index1 = len(sorted_list1) // 2
    middle_index2 = len(sorted_list2) // 2

    print(sorted_list1[middle_index1].hv1)
    print(sorted_list1[middle_index1].res1)
    print(sorted_list2[middle_index2].hv2)
    print(sorted_list2[middle_index2].res2)
    res1 = sorted_list1[middle_index1].res1
    res2 = sorted_list2[middle_index2].res2
    fig, ax = plt.subplots()

    scatter1 = ax.scatter(res1[:, 0], res1[:, 1], color="red", label='Selection', alpha=0.6)
    scatter2 = ax.scatter(res2[:, 0], res2[:, 1], color="blue", label='Weighting', alpha=0.6)

    plt.xlabel('100 - Accuracy')
    plt.ylabel('Remaining Features')
    plt.legend(handles=[scatter1, scatter2])
    plt.show()


if __name__ == "__main__":
    main()