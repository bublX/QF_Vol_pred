import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

class MetricsCalculator:
    def __init__(
        self,
        metrics : list = None
    ):
        self.metrics = metrics or ['MSE', 'RMSE', 'MAE', 'R2']

    def calculate(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray 
    ) -> Dict[str, float]:

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        if len(y_true) != len(y_pred):
            raise ValueError(f'Shapes are different: {len(y_true)} and {len(y_pred)}')

        results = {}

        if 'MSE' in self.metrics:
            results['MSE'] = mean_squared_error(y_true, y_pred)

        if 'RMSE' in self.metrics:
            results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))

        if 'MAE' in self.metrics:
            results['MAE'] = mean_absolute_error(y_true, y_pred)

        if 'R2' in self.metrics:
            results['R2'] = r2_score(y_true, y_pred)

        return results

    def calculate_all(
        self,
        y_true_dict : Dict[str, np.ndarray],
        y_pred_dict : Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:

        results = {}
        for ticker in y_true_dict:
            results[ticker] = self.calculate(
                y_true_dict[ticker],
                y_pred_dict[ticker]
            )

        return results