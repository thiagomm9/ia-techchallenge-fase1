import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def print_validation_metrics(test, predict):
    lin_mae = mean_absolute_error(test, predict)
    r2 = r2_score(test, predict)
    mape = calculate_mape(test, predict)
    
    print(f'Erro Absoluto Médio: {lin_mae}')
    print(f'R² (coeficiente de determinação): {r2}')
    print(f'MAPE (Mean Absolute Percentage Error): {mape}')


def calculate_mape(labels, predictions):
    errors = np.abs(labels - predictions)
    relative_errors = errors / np.abs(labels)
    mape = np.mean(relative_errors) * 100
    return mape