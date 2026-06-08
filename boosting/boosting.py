from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin

import pandas as pd


class Boosting(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        early_stopping_rounds=None,
        eval_metric=None,
        cat_features=None,
        cat_features_ordered=True,
        l2=0,
        loss="BCE",
        focal_gamma=2,
        goss = False,
        goss_k= 0.2,
        subsample = 0.3,
        quantization_type= None,
        nbins= 255,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)
        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.goss = goss
        self.goss_k = goss_k
        self.subsample = subsample
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.bonds = {}
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        if self.loss == "BCE":
            self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
            self.grad_fn = lambda y, z: y / (1 + np.exp(y * z))
            self.hess_fn = lambda y, z: np.exp(y * z) / (1 + np.exp(y * z)) ** 2
        elif self.loss == "FocalLoss":
            self.loss_fn = self.focal_loss
            self.grad_fn = self.focal_grad
        self.iteration = 0
        self.cat_features = cat_features
        self.cat_features_ordered = cat_features_ordered
        self.l2 = l2

    def focal_loss(self, y, z):
        sigm = self.sigmoid(y * z)
        return -(((1 - sigm) ** self.focal_gamma) * np.log(sigm)).mean()

    def focal_grad(self, y, z):
        sigm = self.sigmoid(y * z)
        return y * ((1 - sigm) ** self.focal_gamma) * ((1 - sigm) - self.focal_gamma * sigm * np.log(sigm))
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        iter = self.iteration
        old_predictions = self.train_predictions
        antigrad = self.grad_fn(y, old_predictions)
        model_params = self.base_model_params.copy()
        if self.random_state is not None and self.l2 == 0:
            model_params["random_state"] = self.random_state + iter
        if self.l2 != 0:
            model_params["l2"] = self.l2
        model = self.base_model_class(**model_params)
        if self.l2 != 0:
            hess = self.hess_fn(y, old_predictions)
            if self.goss:
                X_sample, antigrad_sample, hess_sample = self._goss(X, antigrad, hess, iter)
                model.fit(X_sample, antigrad_sample, hess_sample)
            else:
                model.fit(X, antigrad, hess)
            new_predictions = model.predict(X)
            gamma = 1.0
        else:
            if self.goss:
                X_sample, antigrad_sample, _ = self._goss(X, antigrad, None, iter)
                model.fit(X_sample, antigrad_sample)
            else:
                model.fit(X, antigrad)
            new_predictions = model.predict(X)
            gamma = self._find_optimal_gamma(y, old_predictions, new_predictions)
        self.models[iter] = model
        self.gammas[iter] = gamma
        self.train_predictions += self.learning_rate * gamma * new_predictions
        self.history["train_loss"].append(self.loss_fn(y, self.train_predictions))
        self.history["train_roc_auc"].append(roc_auc_score(y == 1, self.sigmoid(self.train_predictions)))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: tuple[np.ndarray, np.ndarray] | None = None, use_best_model: bool = False) -> None:
        if self.cat_features is not None:
            self._cat_fit(X_train, y_train)
            X_train = self._cat_transform(X_train, y_train, ordered=self.cat_features_ordered)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        self._quantization_fit(X_train)
        X_train = self._quantization_transform(X_train)
        self.train_predictions = np.zeros(X_train.shape[0])
        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки
        estimator_range = range(self.n_estimators)
        if eval_set is not None:
            X_valid, y_valid = eval_set
            if self.cat_features is not None:
                X_valid = self._cat_transform(X_valid)
            X_valid = np.asarray(X_valid)
            y_valid = np.asarray(y_valid)
            X_valid = self._quantization_transform(X_valid)
            self.valid_predictions = np.zeros(X_valid.shape[0])
        best_score = None
        best_iteration = 0
        bad = 0
        if self.verbose:
            estimator_range = tqdm(estimator_range)
        for i in estimator_range:
            self.iteration = i
            self.partial_fit(X_train, y_train)
            if eval_set is not None:
                model = self.models[i]
                gamma = self.gammas[i]
                self.valid_predictions += (self.learning_rate * gamma * model.predict(X_valid))
                valid_loss = self.loss_fn(y_valid, self.valid_predictions)
                valid_roc_auc = roc_auc_score(y_valid == 1, self.sigmoid(self.valid_predictions))
                self.history["valid_loss"].append(valid_loss)
                self.history["valid_roc_auc"].append(valid_roc_auc)
                if self.eval_metric == "valid_roc_auc":
                    score = valid_roc_auc
                    if best_score is not None:
                        is_better = score > best_score
                    else:
                        is_better = True
                else:
                    score = valid_loss
                    if best_score is not None:
                        is_better = score < best_score
                    else:
                        is_better = True
                if is_better:
                    best_score = score
                    best_iteration = i
                    bad = 0
                else:
                    bad += 1
                if self.early_stopping_rounds is not None and bad == self.early_stopping_rounds:
                    break
        # чтобы было удобнее смотреть
        if use_best_model:
            self.models = self.models[:best_iteration + 1]
            self.gammas = self.gammas[:best_iteration + 1]
        for key in self.history:
            self.history[key] = np.array(self.history[key])
    
    def _cat_fit(self, X, y):
        y = pd.Series(y == 1)
        self.cat_means = {}
        for feature in self.cat_features:
            self.cat_means[feature] = y.groupby(X[feature]).mean()
    
    def _cat_transform(self, X, y=None, ordered=False):
        X = X.copy()
        if ordered:
            y = pd.Series(y == 1)
            for feature in self.cat_features:
                pref_sum = y.groupby(X[feature]).cumsum() - y
                pref_count = X[feature].groupby(X[feature]).cumcount()
                X[feature] = pref_sum / pref_count
                X[feature] = X[feature].fillna(0.5)
            return X
        for feature in self.cat_features:
            X[feature] = X[feature].map(self.cat_means[feature])
            X[feature] = X[feature].fillna(0.5)
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.cat_features is not None:
            X = self._cat_transform(X)
        X = np.asarray(X)
        X = self._quantization_transform(X)
        predictions = np.zeros(X.shape[0])
        for model, gamma in zip(self.models, self.gammas):
            if model != 0:
                predictions += self.learning_rate * gamma * model.predict(X)
        return np.stack([1 - self.sigmoid(predictions), self.sigmoid(predictions)], axis=1)

    def _find_optimal_gamma(self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]
    
    def _goss(self, X: np.ndarray, antigrad, hess=None, iter=0):
        if not self.goss:
            return X, antigrad, hess
        k, s = self.goss_k, self.subsample
        top = max(1, int(k * len(antigrad)))
        order = np.argsort(-np.abs(antigrad))
        big = order[:top]
        rest = order[top:]
        if self.random_state is None:
            np.random.seed(self.random_state + iter)
        rest_chosen = rest[np.random.rand(len(rest)) < s]
        ind = np.concatenate([big, rest_chosen])
        ag_sub = antigrad[ind].copy()
        ag_sub[top:] *= 1.0 / s
        if hess is not None:
            hess_sub = hess[ind].copy()
            hess_sub[top:] *= 1.0 / s
        else:
            hess_sub = None
        return X[ind], ag_sub, hess_sub
    
    def _quantization_fit(self, X):
        if self.quantization_type is None:
            return
        self.bonds = {}
        for feature in range(X.shape[1]):
            vals = X[:, feature]
            if len(np.unique(vals)) <= 1:
                self.bonds[feature] = np.array([])
                continue
            if self.quantization_type == 'uniform':
                bound = np.linspace(np.min(vals), np.max(vals), self.nbins + 1)[1:-1]
            elif self.quantization_type == 'quantile':
                quantiles = np.linspace(0, 1, self.nbins + 1)[1:-1]
                bound = np.quantile(vals, quantiles)
                bound = np.unique(bound)
            self.bonds[feature] = bound

    def _quantization_transform(self, X):
        if self.quantization_type is None:
            return X
        X_quant = np.zeros(X.shape)
        for feature in range(X.shape[1]):
            bonds = self.bonds[feature]
            X_quant[:, feature] = np.searchsorted(bonds, X[:, feature], side='right')
        return X_quant

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])
    
    def plot_history(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            plt.plot(self.history[key], label=key)
        plt.xlabel("iteration")
        plt.ylabel("metric")
        plt.legend()
        plt.grid()
        plt.show()
        
    def get_feature_importance(self, X=None, y=None, type="split"):
        importances = np.zeros(self.models[0].n_features_in_)
        for model, gamma in zip(self.models, self.gammas):
            importances += gamma * model.feature_importances_
        return importances / importances.sum()
