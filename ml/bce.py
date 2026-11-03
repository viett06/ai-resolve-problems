
import math

def binary_cross_entropy(y_true, y_prob):
    epsilon = 1e-15
    total_loss = 0
    n = len(y_true)

    for i in range(n):
        p = y_prob[i]

        # Prevent log(0)
        p = max(epsilon, min(1 - epsilon, p))

        y = y_true[i]

        loss = -(
            y * math.log(p)
            + (1 - y) * math.log(1 - p)
        )

        total_loss += loss

    return total_loss / n
