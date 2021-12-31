import numpy as np
import xgboost as xgb
from river.drift import PageHinkley


class AdaptiveXGBoostRegressor():
    _PUSH_STRATEGY = 'push'
    _REPLACE_STRATEGY = 'replace'
    _UPDATE_STRATEGIES = [_PUSH_STRATEGY, _REPLACE_STRATEGY]

    def __init__(self,
                 n_estimators=30,
                 learning_rate=0.3,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None,
                 update_strategy='replace',
                 detect_drift=False,
                 threshold=200,
                 min_instances=200,
                 delta=0.5):

        """
        Adaptive XGBoost regressor.
        Parameters
        ----------
        n_estimators: int (default=5)
            The number of estimators in the ensemble.
        learning_rate: float (default=0.3)
            Learning rate, a.k.a eta.
        max_depth: int (default = 6)
            Max tree depth.
        max_window_size: int (default=1000)
            Max window size.
        min_window_size: int (default=None)
            Min window size. If this parameters is not set, then a fixed size
            window of size ``max_window_size`` will be used.
        update_strategy: str (default='replace')
            | The update strategy to use:
            | 'push' - the ensemble resembles a queue
            | 'replace' - oldest ensemble members are replaced by newer ones
        detect_drift: bool (default=False)
            If set will use a drift detector (Page-Hinckley).
        threshold: int (default=200)
            The change is detected if the difference is greater than a threshold (lambda)
        min_instances: int (default=200)
            Minimum number of instances before detecting change.
        delta: float (default=0.5)
            Magnitude of changes that are allowed


        Notes
        -----
        The Adaptive XGBoost [1]_ (AXGB) regressor is an adaptation of the
        XGBoost algorithm for evolving data streams. AXGB creates new members
        of the ensemble from mini-batches of data as new data becomes
        available.  The maximum ensemble  size is fixed, but learning does not
        stop once this size is reached, the ensemble is updated on new data to
        ensure consistency with the current data distribution.
        References
        ----------
        .. [1] Montiel, Jacob, Mitchell, Rory, Frank, Eibe, Pfahringer,
           Bernhard, Abdessalem, Talel, and Bifet, Albert. “AdaptiveXGBoost for
           Evolving Data Streams”. In:IJCNN’20. International Joint Conference
           on Neural Networks. 2020. Forthcoming.
        """
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
        self._samples_seen = 0
        self.threshold = threshold
        self.min_instances = min_instances
        self.delta = delta
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
        self._boosting_params = {"verbosity": 1,
                                 "objective": "reg:squarederror",
                                 "eta": self.learning_rate,
                                 "max_depth": self.max_depth}
        if self.detect_drift:
            self._drift_detector = PageHinkley(threshold=self.threshold, min_instances=self.min_instances, delta=self.delta)
       

    def reset(self):
        """
        Reset the estimator.
        """
        self._first_run = True
        self._configure()

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
        AdaptiveXGBoostRegressor
            self
        """
        for i in range(X.shape[0]):
            self._partial_fit(np.array([X[i, :]]), np.array([y[i]]))
        return self

    def _partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, np.shape(X)[1])
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
            error = abs(self.predict(X)-y)
            # Check for warning
            in_drift, in_warning = self._drift_detector.update(error)
            # Check if there was a change
            if in_drift:
                print(f'A concept drift of {self.update_strategy} strategy was detected in the position: {self._samples_seen}')
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
            self._samples_seen += X.shape[0]
            self._update_model_idx()
        else:   # self.update_strategy == self._PUSH_STRATEGY
            booster = self._train_booster(X, y, len(self._ensemble))
            # Update ensemble
            if len(self._ensemble) == self.n_estimators:
                self._ensemble.pop(0)
            self._ensemble.append(booster)
            self._samples_seen += X.shape[0]

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
        Predict the target for sample X
        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the samples to
            predict the target for.
        Returns
        -------
        numpy.ndarray
            A 1D array of shape (, n_samples), containing the
            predicted target for all instances in X.
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
        return np.zeros(np.shape(X)[0])

