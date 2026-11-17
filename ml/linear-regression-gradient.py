def linear_gradient(xs, ys, weight, bias):
    n = len(xs)

    dw = 0
    db = 0
    for i in range(n):
        y_pred = weight * xs[i] + bias

        error = y_pred - ys[i]
        dw += error * xs[i]
        db += error

    dw = (2/n) * dw
    db = (2/n) * db
    return [dw, db]