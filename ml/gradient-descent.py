def gradient_step(parameters, gradients, learning_rate):
    update_params = []

    for i in range(len(parameters)):
        new_value = (
            parameters[i] - learning_rate * gradients[i]
        )
        update_params.append(new_value)

    return update_params
