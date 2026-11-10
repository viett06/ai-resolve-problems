import math

def sigmoid(xs):
    results = []

    for x in xs:
        if x >= 0:
            value = 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            value = exp_x / (1 + exp_x)

        results.append(value)

    return results
