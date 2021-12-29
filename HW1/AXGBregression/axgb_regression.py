import numpy as np
import xgboost as xgb
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.utils import get_dimensions

class AdaptiveXGBoostClassifier():
    _PUSH_STRATEGY = 'push'
    _REPLACE_STRATEGY = 'replace'
    _UPDATE_STRATEGIES = [_PUSH_STRATEGY, _REPLACE_STRATEGY]

    def __init__(self,
                 n_estimators=30,
                 learning_rate=1,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None,
                 detect_drift=False,
                 update_strategy='replace',
                 threshold = 100):

        super().__init__()
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self._first_run = True
        self._ensemble = None
        self.detect_drift = detect_drift
        self._drift_detector = None
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])
        self.threshold = threshold
        self._model_idx = 0
        if update_strategy not in self._UPDATE_STRATEGIES:
            raise AttributeError("Invalid update_strategy: {}\n"
                                 "Valid options: {}".format(update_strategy,
                                                            self._UPDATE_STRATEGIES))
        self.update_strategy = update_strategy
        self._configure()

    def _configure(self):
        if self.update_strategy == self._PUSH_STRATEGY:
            self._ensemble = []
        elif self.update_strategy == self._REPLACE_STRATEGY:
            self._ensemble = [None] * self.n_estimators
        self._reset_window_size()
        self._init_margin = 0.0
        self._boosting_params = {"verbosity": 0,
                                 "objective": "reg:squarederror",
                                 "eta": self.learning_rate,
                                 "max_depth": self.max_depth}
        if self.detect_drift:
            self._drift_detector = PageHinkley( threshold=self.threshold,min_instances = 200, alpha = 0.5)


    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Partially (incrementally) fit the model.
        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the data upon which
            the algorithm will create its model.
        y: Array-like
            An array of shape (, n_samples) containing the classification
            targets for all samples in X. Only binary data is supported.
        classes: Not used.
        sample_weight: Not used.
        Returns
        -------
        AdaptiveXGBoostClassifier
            self
        """
        for i in range(X.shape[0]):
            self._partial_fit(np.array([X[i, :]]), np.array([y[i]]))
        return self

    def _partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_buffer = np.array([])
            self._first_run = False
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))
        while self._X_buffer.shape[0] >= self.window_size:
            self._train_on_mini_batch(X=self._X_buffer[0:self.window_size, :],
                                      y=self._y_buffer[0:self.window_size])
            delete_idx = [i for i in range(self.window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)

            # Check window size and adjust it if necessary
            self._adjust_window_size()

        # Support for concept drift
        if self.detect_drift:
            error = abs (self.predict(X)-y)
            # Check for warning
            self._drift_detector.add_element(error)
            # Check if there was a change
            if self._drift_detector.detected_change():
                # Reset window size
                self._reset_window_size()
                if self.update_strategy == self._REPLACE_STRATEGY:
                    self._model_idx = 0

    def _adjust_window_size(self):
        if self.window_size * 2 < self.max_window_size:
            self.window_size *= 2
        else:
            self.window_size = self.max_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self.window_size = self.min_window_size
        else:
            self.window_size = self.max_window_size

    def _train_on_mini_batch(self, X, y):
        if self.update_strategy == self._REPLACE_STRATEGY:
            booster = self._train_booster(X, y, self._model_idx)
            # Update ensemble
            self._ensemble[self._model_idx] = booster
            self._update_model_idx()
        else:
            booster = self._train_booster(X, y, len(self._ensemble))
            # Update ensemble
            if len(self._ensemble) == self.n_estimators:
                self._ensemble.pop(0)
            self._ensemble.append(booster)

    def _train_booster(self, X: np.ndarray, y: np.ndarray, last_model_idx: int):
        d_mini_batch_train = xgb.DMatrix(X, y)
        # Get margins from trees in the ensemble
        margins = np.asarray([self._init_margin] * d_mini_batch_train.num_row())
        for j in range(last_model_idx):
            margins = np.add(margins,
                             self._ensemble[j].predict(d_mini_batch_train, output_margin=True))
        d_mini_batch_train.set_base_margin(margin=margins)
        booster = xgb.train(params=self._boosting_params,
                            dtrain=d_mini_batch_train,
                            num_boost_round=1,
                            verbose_eval=False)
        return booster

    def _update_model_idx(self):
        self._model_idx += 1
        if self._model_idx == self.n_estimators:
            self._model_idx = 0

    def predict(self, X):
        """
        Predict the class label for sample X
        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the samples to
            predict the class label for.
        Returns
        -------
        numpy.ndarray
            A 1D array of shape (, n_samples), containing the
            predicted class labels for all instances in X.
        """
        if self._ensemble:
            if self.update_strategy == self._REPLACE_STRATEGY:
                trees_in_ensemble = sum(i is not None for i in self._ensemble)
            else:   # self.update_strategy == self._PUSH_STRATEGY
                trees_in_ensemble = len(self._ensemble)
            if trees_in_ensemble > 0:
                d_test = xgb.DMatrix(X)
                for i in range(trees_in_ensemble - 1):
                    margins = self._ensemble[i].predict(d_test, output_margin=True)

                    d_test.set_base_margin(margin=margins)
                predicted = self._ensemble[trees_in_ensemble - 1].predict(d_test)

                return np.array(predicted).astype(float)
        # Ensemble is empty, return default values (0)
        return np.zeros(get_dimensions(X)[0])

