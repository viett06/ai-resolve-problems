def mse(y_true, y_pred):
    n = len(y_true)
    total_error = 0

    for i in range(n):
        error = y_true[i] - y_pred[i]
        squared_error = error ** 2
        total_error += squared_error

    return total_error / n


def main():
    from math import isclose

    test_cases = [
        ([3, -0.5, 2, 7], [2.5, 0.0, 2, 8], 0.375),
        ([1, 2, 3], [1, 2, 3], 0.0),
        ([5], [2], 9.0),
    ]

    for index, (y_true, y_pred, expected) in enumerate(test_cases, start=1):
        result = mse(y_true, y_pred)
        if not isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(
                f"Test {index} failed: expected {expected}, got {result}"
            )
        print(f"Test {index} PASS: mse({y_true}, {y_pred}) = {result}")

    print("All tests passed!")


if __name__ == '__main__':
    main()
