# Multi Objective Optimization Project

This repository contains the implementation of a multi-objective optimization project focused on maximizing the accuracy of the model and minimizing the remaining features. The project uses the Non-dominated Sorting Genetic Algorithm II (NSGA2) for optimization and compares the performance of feature selection to feature weighting.

## Sampling
The Sampling class is responsible for generating the initial population for the genetic algorithm. This project uses a custom sampling strategy for both the weighting and selection methods. For the weighting method, the initial population is populated with values ranging between 0.5 and 1. A certain number of these values are then randomly altered to fall below 0.5. On the other hand, for the selection method, the initial population is initialized with all values set to 1. Again, a number of these values are then randomly selected and converted to 0.

## Mutation
Mutation is a genetic operator used to maintain diversity within the population. It alters one or more gene values in a chromosome from its initial state. This project introduces an alternative approach where randomly generated weights are aligned towards zero using the random.exponential function. This strategy aims to reduce the number of remaining features in subsequent generations, thus enhancing the efficiency of the optimization process.

## Problem

The problem class encapsulates the specific problem that the genetic algorithm is trying to solve. This project uses a custiom Problem class which inherits from the Problem class provided by PyMOO. For each solution in the population (a list of weights for each feature), it creates a new weighted version of the dataset by multiplying the original dataset with these weights. It then splits this weighted dataset into training and testing sets, fits a k-nearest neighbors (KNN) model to the training data, and makes predictions on the test data. The accuracy of these predictions is calculated and stored. The method also counts the number of features that have been given a weight of zero. These two metrics (accuracy and count of zero weights) form the objective values (F) for the solution.
