
# Constraint Optimization using Penalty Function Method

## Overview

The goal in every industry is to minimize or maximize a particular quantity. In the real world, there are generally multiple variables involved in the problem. To achieve this goal, optimization methods were introduced. These methods help us in narrowing down the search area for a particular optimum solution.

This repository contains code for finding the optimum points of three different objective functions, each with several constraints. The implementation uses the Penalty Function Method for constraint optimization on multivariable functions.

## Methods Employed

- **Penalty Function Method**: Used for constraint optimization on multivariable functions.
- **Bounding Phase Method**: A technique to identify the interval containing the optimum point.
- **Interval Halving Method**: A method to iteratively reduce the interval size and home in on the optimum point.
- **Conjugate Gradient Method**: An efficient algorithm for large-scale unconstrained optimization problems.

## Implementation

With the exception of `matplotlib` for plotting, all of the methods were indigenously developed in Python. The repository includes scripts and functions that implement these optimization techniques and apply them to various objective functions with different constraints.

## Requirements

- Python 3.x
- `matplotlib` for plotting

## Usage

Clone the repository and navigate to the project directory. Run the scripts to see the optimization methods in action and visualize the results.

```bash
python main.py
```


## Authors

- [Koushik Samanta](https://www.linkedin.com/in/samanta-koushik/)
- [Sourabh Singh](https://www.linkedin.com/in/sourabhsingh/)
