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


from skmultiflow.evaluation import EvaluatePrequential

# Adaptive XGBoost classifier parameters
n_estimators = 5       # Number of members in the ensemble
learning_rate = 0.3     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 1000  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = False    # Enable/disable drift detection

AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift)
AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift)

stream = RegressionGenerator (n_samples = 20000, 
        n_features = 15,
        random_state=100)

evaluator = EvaluatePrequential(pretrain_size=0,
                                max_samples=20000,
                                metrics=['mean_square_error'],
                                show_plot=False)



evaluator.evaluate(stream=stream,
                   model=[AXGBp, AXGBr],
                  model_names=['AXGBp', 'AXGBr'])