import numpy as np
import math
import random
import sys
import matplotlib.pyplot as plt

x_series = []


class Objective:
    dictForObjectiveFunction = {}
    noOfFunctionEvaluations = 0
    penaltyFactor = 0
    noOfVarAllProblem = [2, 0, 0, 2]

    def __init__(self, problemIndicator) -> None:
        self.problemIndicator = problemIndicator
        self.noOfVarCurProblem = self.noOfVarAllProblem[problemIndicator-1]

    def objectiveFunction(self, *x):
        # check whether values are already stored or not.
        if x in self.dictForObjectiveFunction:
            return self.dictForObjectiveFunction.get(x)

        # when values are not stored calculate fresh.
        result = 0

        # Problem 1
        if self.problemIndicator == 1:
            result = (x[0]-10)**3+(x[1]-20)**3
        elif self.problemIndicator == 2:
            pass
        elif self.problemIndicator == 3:
            pass
        elif self.problemIndicator == 4:
            # test case
            result = (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
        else:
            pass

        # store the new calculated value in the dictionary
        self.dictForObjectiveFunction[x] = result
        self.noOfFunctionEvaluations += 1
        return result

    def equilityConstraint(self, *x):
        return []

    def gtrInEquilityConstraint(self, *x):
        constraintViolation = []
        if self.problemIndicator == 1:
            constraintViolation.append(
                (x[0]-5)**2+(x[1]-5)**2-100
            )

            # Also for variable lower bounds
            constraintViolation.append(x[0]-13)
            constraintViolation.append(x[1]-0)

        elif self.problemIndicator == 4:
            constraintViolation.append(
                (x[0]-5)**2+x[1]**2-26
                )
            constraintViolation.append(x[0]-0)
            constraintViolation.append(x[1]-0)
        else:
            pass
        return constraintViolation

    def lwrInwquilityConstraint(self, *x):
        constraintViolation = []
        if self.problemIndicator == 1:
            constraintViolation.append(
                (x[0]-6)**2+(x[1]-5)**2-82.81
            )
            # Also for variable upper bounds
            constraintViolation.append(x[0]-20)
            constraintViolation.append(x[1]-4)
        else:
            pass
        return constraintViolation

    def penaltyFunction(self, *x):
        penaltyTerm = 0
        for m in self.gtrInEquilityConstraint(*x):
            if m < 0:
                penaltyTerm = penaltyTerm + self.penaltyFactor*(m**2)
            else:
                pass
        for n in self.lwrInwquilityConstraint(*x):
            if n > 0:
                penaltyTerm = penaltyTerm + self.penaltyFactor*(n**2)
            else:
                pass
        for o in self.equilityConstraint(*x):
            pass

        return self.objectiveFunction(*x)+penaltyTerm


def partialDerivative(functionToOperate, variableIndicator, currentPoint):
    """
    This function will partially derive the a function
    with respect to variabel at a given point.
    It uses central difference method to implement the partial derivatives of first order.

    Args:
        functionToOperate (call back function): [function on which we will be differentiating]
        variableIndicator (int): [its an indicator for variable, starts from 1, with respect to which we will be partially differentiating]
        currentPoint (list): [current point at which we need to make the differentiation]

    Returns:
        [number]: [value]
    """
    deltaX = 10**-4
    pointOne = currentPoint.copy()
    pointTwo = currentPoint.copy()
    indicatorValue = currentPoint[variableIndicator-1]
    pointOne[variableIndicator-1] = indicatorValue + deltaX
    pointTwo[variableIndicator-1] = indicatorValue - deltaX
    return (functionToOperate(*pointOne)-functionToOperate(*pointTwo))/(2*deltaX)


def gradiantOfFunction(functionToOperate, currentPoint):
    """Generate gradiant of a vector at a particular point

    Args:
        functionToOperate (call back function): function on which gradiant to be operate on.
        currentPoint (list): current point at which gradiant to be calculated.

    Returns:
        numpy array: gradiant vector
    """

    # Create a Zero matrix with no. of rows = no of variable in currentpoint
    # and only one column
    A = np.zeros((len(currentPoint), 1))
    for i in range(len(currentPoint)):
        A[i][0] = partialDerivative(functionToOperate, i+1, currentPoint)

    return A


def boundingPhaseMethod(functionToOperate, delta, a, b):
    """This is a Bracketing method.
    Which will be used to optimize a single variable function.

    Args:
        functionToOperate (call back function): Objective Function
        delta (float): Separation between two points
        a (float): Lower limit
        b (float): Upper limit

    Returns:
        [a,b]: List containing the brackets [Lower,Upper].
    """

    deltaWithSign = None
    k = 0
    maxIteration = 10000
    initialRange = [a,b]
    while True:
        # step 1
        x_0 = random.uniform(a, b)
        if (x_0 == a or x_0 == b):
            continue

        # step 2
        # In the below code there will be 3 function evaluations
        if functionToOperate(x_0 - abs(delta)) >= functionToOperate(x_0) and functionToOperate(x_0 + abs(delta)) <= functionToOperate(x_0):
            deltaWithSign = + abs(delta)
        elif functionToOperate(x_0 - abs(delta)) <= functionToOperate(x_0) and functionToOperate(x_0 + abs(delta)) >= functionToOperate(x_0):
            deltaWithSign = - abs(delta)
        else:
            continue

        while True:
            if k >= maxIteration:
                return initialRange
                #sys.exit("From Bounding Phase Method : No optimal point found")

            # step 3
            x_new = x_0 + 2**k*deltaWithSign
            if functionToOperate(x_new) < functionToOperate(x_0):
                k += 1
                x_0 = x_new
                continue
            else:
                # return in [x_lower,x_upper] format
                temp1 = x_new-(2**k)*1.5*deltaWithSign
                temp2 = x_new
                if temp1 > temp2:
                    return [temp2, temp1]
                else:
                    return [temp1, temp2]

    '''
    # The total no of function evaluation should be equal to (No. of iteration + 2)
    for this function as k starts with 0 so the total no of function evaluation = (k+1)+2
    '''


def intervalHalvingMethod(functionToOperate, epsinol, a, b):
    """This is a Region Elimination method.
    Which will be used to find the optimal solution for a single variable function.

    Args:
        functionToOperate (call back function): Objective Function
        epsinol (float): Very small value used to terminate the iteration
        a (float): Lower limit
        b (float): Upper limit

    Returns:
        List: A list contains the bracket [lower,upper]
    """

    # step 1
    maxIteration = 10000
    x_m = (a+b)/2
    l = b-a
    no_of_iteration = 1
    initialRange = [a,b]
    while True:
        if no_of_iteration >= maxIteration:
            return initialRange
            #sys.exit("From Bounding Phase Method : No optimal point found")
        # step2
        x_1 = a+l/4
        x_2 = b-l/4
        while True:
            # step3
            if functionToOperate(x_1) < functionToOperate(x_m):
                b = x_m
                x_m = x_1
                break

            # step4
            if functionToOperate(x_2) < functionToOperate(x_m):
                a = x_m
                x_m = x_2
                break

            else:
                a = x_1
                b = x_2
                break

        # step5
        l = b-a
        if abs(l) < epsinol:
            return [a, b]
        else:
            no_of_iteration += 1
            continue


def changeToUniDirectionFunction(functionToOperate, x, s):
    """This is a function which will be used to scale the multivariable function
    into a single variable function using uni direction method.

    Args:
        functionToOperate (call back function): Multivariable objective function
        x (row vector): Position of initial point in [1,2,...] format
        s (row vector): Direction vector in [1,2,....] format

    Returns:
        function: Scaled functiion in terms of single variable -> a
    """
    return lambda a: functionToOperate(*(np.array(x)+np.multiply(s, a)))


def conjugateGradiantMethod(functionToOperate, limits, initialPoint):
    """This is an Gradiant Based Multi-Variable Optimisation Method.
    It is used to find the optimal solution of an objective function.

    Args:
        functionToOperate (call back function): Objective Function
        limits (list): in [lower,upper] format
        initialPoint (list): in [x0,x1,x2....] format
    """
    angleForDependencyInDegree = 1
    # step 1
    a, b = limits
    x_0 = list(initialPoint)
    epsinolOne = 10**-8
    epsinolTwo = 10**-8
    epsinolThree = 10**-5
    k = 0
    M = 1000
    x_series = []  # store the x vextors
    x_series.append(x_0)

    # step2
    s_series = []  # store the direction vectors
    gradiantAtX_0 = gradiantOfFunction(functionToOperate, x_0)
    s_series.append(-gradiantAtX_0)
    # print(x_series[-1],gradiantAtX_0)
    # Extra termination condition *****
    if (np.linalg.norm(gradiantAtX_0)) <= epsinolThree:
        print(f"CG: Termination Point 1. Iterations Count -> {k}")
        return (x_0)

    # step3
    # convert multivariable function into single variable function
    s_0 = s_series[0][:, 0]
    newObjectiveFunction = changeToUniDirectionFunction(
        functionToOperate, x_0, s_0)

    # search for unidirection optimal point
    m, n = boundingPhaseMethod(newObjectiveFunction, 10**-1, a, b)
    m, n = intervalHalvingMethod(newObjectiveFunction, epsinolOne, m, n)
    optimumPoint = (m+n)/2
    # print("Optimum Point",optimumPoint)

    x_1 = (np.array(x_0)+np.multiply(s_0, optimumPoint))
    # print(x_1)
    x_series.append(x_1)
    k = 1

    while(True):

        # step 4
        part_1_s = -gradiantOfFunction(functionToOperate, x_series[k])
        p = (np.linalg.norm(-part_1_s))**2
        q = gradiantOfFunction(functionToOperate, x_series[k-1])
        r = (np.linalg.norm(q))**2
        t = p/r
        part_2_s = np.multiply(s_series[k-1], t)
        s = part_1_s + part_2_s
        s_series.append(s)  # s_series size will become to k+1

        # code to check linear independance
        s_k = s_series[k][:, 0]  # row vector
        s_k_1 = s_series[k-1][:, 0]  # row vector
        dotProduct = s_k @ s_k_1
        factor = np.linalg.norm(s_k)*np.linalg.norm(s_k_1)
        finalDotProduct = dotProduct/factor
        finalDotProduct = round(finalDotProduct, 3)
        dependencyCheck = math.acos(
            finalDotProduct)*(180/math.pi)  # in degrees
        # print(dependencyCheck)
        if abs(dependencyCheck) < angleForDependencyInDegree:
            # Restart
            print(f"Linear Dependency Found! Restarting with {x_series[k]}")
            return conjugateGradiantMethod(functionToOperate, limits, x_series[k])

        # step 5
        # convert multivariable function into single variable function
        x_k = x_series[k]  # 1-D list
        s_k = s_series[k][:, 0]  # 1-D list
        newObjectiveFunction = changeToUniDirectionFunction(
            functionToOperate, x_k, s_k)

        # search for unidirection optimal point
        m, n = boundingPhaseMethod(newObjectiveFunction, 10**-1, a, b)
        m, n = intervalHalvingMethod(newObjectiveFunction, epsinolOne, m, n)
        optimumPoint = (m+n)/2
        x_new = (np.array(x_k)+np.multiply(s_k, optimumPoint))
        x_series.append(x_new)  # x_series size will be k+2

        # step 6
        # check the terminate condition
        norm_1 = np.linalg.norm(np.array(x_series[k+1])-np.array(x_series[k]))
        norm_2 = np.linalg.norm(x_series[k])
        factor = np.linalg.norm(gradiantOfFunction(
            functionToOperate, x_series[k+1]))

        if norm_2 != 0:
            if norm_1/norm_2 <= epsinolTwo:
                print(f"CG: Termination Point 2. Iterations Count -> {k}")
                return x_series[k+1]

        if factor <= epsinolThree or k+1 >= M:
            # terminate the function
            print(f"CG: Termination Point 3. Iterations Count -> {k+1}")
            return x_series[k+1]
        else:
            k += 1
            continue

        break


def penaltyFunctionMethod(object, limits, initialPoint):
    # step 1
    global x_series
    epsinol = 10**-5
    object.penaltyFactor = 0.1
    c = 10
    k = 0
    x_series.append(initialPoint)

    while(True):
        # step2
        # should be passed as an object->'object' to this method

        # step 3
        x_new = conjugateGradiantMethod(
            object.penaltyFunction, limits, x_series[k])

        # step 4
        term_1 = object.penaltyFunction(*x_new)
        if k != 0:
            # calculate the penalty with the previous value of penalty factor
            object.penaltyFactor = object.penaltyFactor/c
            term_2 = object.penaltyFunction(*x_series[k])
            # again reset the penalty factor as it is
            object.penaltyFactor = object.penaltyFactor*c
        else:
            term_2 = object.penaltyFunction(*x_series[k])
        x_series.append(x_new)

        # termination condition
        if abs(term_1-term_2) <= epsinol:
            return x_new

        # step 5
        object.penaltyFactor = object.penaltyFactor*c
        k = k+1


p1 = Objective(4)
print(penaltyFunctionMethod(p1,[0,10],[0,0]))