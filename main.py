from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from problem import OptimisationProblem
from sampling import CustomSampling, CustomSamplingMax


def main():
    pop_size = 50
    algorithm = NSGA2(pop_size=pop_size, sampling=CustomSampling())
    problem = OptimisationProblem()
    res = minimize(problem, algorithm, ('n_gen', 10), seed=1, verbose=False)

    # Plot the results
    print(res.F)
    plot = Scatter()
    plot.add(res.F, color="red")
    plot.show()


if __name__ == "__main__":
    main()