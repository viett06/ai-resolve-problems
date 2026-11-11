
import math

def sigmoid_derivative(xs):
    results = []

    for x in xs:
        # Stable sigmoid
        if x >= 0:
            s = 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            s = exp_x / (1 + exp_x)

        # Sigmoid derivative
        derivative = s * (1 - s)

        results.append(derivative)

    return results
