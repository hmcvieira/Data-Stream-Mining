# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:39:20 2021

@author: hmcarvalhovieira
"""

"""
Created on Wed Nov 17 18:28:41 2021

@author: hmcarvalhovieira
"""

from axgb_regression import AdaptiveXGBoostClassifier

from skmultiflow.data import RegressionGenerator

import matplotlib.pyplot as plt
from skmultiflow.evaluation import EvaluatePrequential

# Adaptive XGBoost classifier parameters
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.3     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 1000  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = True    # Enable/disable drift detection
threshold = 1000 #pagehinkley threshold
AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift,
                                  threshold = threshold)
AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift,
                                  threshold = threshold)

stream = RegressionGenerator (n_samples = 20000, 
        n_features = 15,
        random_state=100)

evaluator = EvaluatePrequential( pretrain_size=0,
                                max_samples=10000,
                                metrics=['mean_square_error', 'running_time'],
                                output_file = 'file_test',
                                show_plot=True)

plt.show()


evaluator.evaluate(stream=stream,
                   model=[ AXGBp],
                  model_names=[ 'AXGBp'])

from skmultiflow.data.file_stream import FileStream





