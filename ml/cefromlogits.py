
import math

def cross_entropy_from_logits(logits, target_index):
    # Step 1: Find max logit
    max_logit = max(logits)

    # Step 2: Shift logits for numerical stability
    shifted = [x - max_logit for x in logits]

    # Step 3: Compute exponentials
    exp_values = [math.exp(x) for x in shifted]

    # Step 4: Sum exponentials
    total = sum(exp_values)

    # Step 5: Compute log-sum-exp
    log_sum_exp = math.log(total)

    # Step 6: Get shifted target logit
    target_logit = shifted[target_index]

    # Step 7: Cross-Entropy Loss
    loss = log_sum_exp - target_logit

    return loss
