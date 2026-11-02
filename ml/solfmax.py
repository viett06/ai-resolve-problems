import math

def softmax(logits):
    max_logit = max(logits)

    # Step 1: Numerical stability
    shifted_logits = [
        x - max_logit for x in logits
    ]

    # Step 2: Exponential
    exp_values = [
        math.exp(x) for x in shifted_logits
    ]

    # Step 3: Sum exponential values
    total = sum(exp_values)

    # Step 4: Normalize to probabilities
    probabilities = [
        value / total for value in exp_values
    ]

    return probabilities
