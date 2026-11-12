
def linear_predict(xs, weight, bias):
    predictions = []

    for x in xs:
        y_pred = weight * x + bias
        predictions.append(y_pred)

    return predictions
