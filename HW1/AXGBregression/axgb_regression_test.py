from axgb_regression import AdaptiveXGBoostRegressor
from skmultiflow.data import RegressionGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream
import numpy as np
import pandas as pd

# Adaptive XGBoost regressor parameters
n_estimators = 20       # Number of members in the ensemble
learning_rate = 0.3     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 100   # Max window size
min_window_size = 10    # set to activate the dynamic window strategy
detect_drift = True     # Enable/disable drift detection
threshold = 5100         # Page-Hinkley threshold


#%% Initializing the models

AXGBp = AdaptiveXGBoostRegressor(update_strategy='push',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift,
                                  threshold=threshold)
AXGBr = AdaptiveXGBoostRegressor(update_strategy='replace',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift,
                                  threshold=threshold)

#%% Generating the dataset
stream_1 = RegressionGenerator(n_samples=2000, n_features=15, n_informative=8, random_state=5)
stream_2 = RegressionGenerator(n_samples=3000, n_features=15, n_informative=10, random_state=5)
stream_3 = RegressionGenerator(n_samples=5000, n_features=15, n_informative=12, random_state=3)
stream = np.concatenate((stream_1, stream_2, stream_3), axis=None)

X_1, y_1 = stream_1.next_sample(2000)
X_2, y_2 = stream_2.next_sample(3000)
X_3, y_3 = stream_3.next_sample(5000)

X = np.concatenate((X_1, X_2, X_3))
y = np.concatenate((y_1, y_2, y_3))

df = pd.DataFrame(np.hstack((X,np.array([y]).T)))
df.to_csv("datasets/file_reg_generator.csv", index=False)
stream = FileStream('datasets/file_reg_generator.csv')

#%% Prequential evaluation

evaluator = EvaluatePrequential(pretrain_size=0,
                                max_samples=10000,
                                metrics=['mean_square_error', 'running_time'],
                                show_plot=True)

evaluator.evaluate(stream=stream,
                   model=[AXGBp, AXGBr],
                   model_names=['AXGBp', 'AXGBr'])


