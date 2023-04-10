def accuracy_all(y_true: list[list[str]], y_hat: list[list[str]]):
    accuracy = [a == b for i in range(len(y_hat)) for (a, b) in zip(y_hat[i], y_true[i])]
    return sum(accuracy) / len(accuracy)