import numpy as np 
import scipy as sp
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC



class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

        

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        errors = X @ w - y
        return float((errors @ errors) / errors.shape[0])


    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        errors = X @ w - y
        return (2 / errors.shape[0]) * (X.T @ errors)
        
    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса; 
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)
        
    
    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        return np.linalg.solve(X.T @ X, X.T @ y)

    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        n, m = X.shape
        k = min(m, n) - 1
        if m == 0:
            return np.zeros(0)
        if n == 0:
            return np.zeros(m)
        if m == 1:
            x = X[:, 0]
            x_norm = np.linalg.norm(x)
            if x_norm == 0.0:
                return np.array([0.0])
            return np.array([(x @ y) / x_norm])
        if n == 1:
            x = X[0, :]
            x_norm = np.linalg.norm(x)
            if x_norm == 0.0:
                return np.zeros(m)
            return (y[0] / x_norm) * x
        U, Sigma, V_T = sp.sparse.linalg.svds(X, k=k)
        w = (V_T.T * (1 / Sigma)) @ U.T @ y
        return w


class L2Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0,
                 analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate
        self.analytic_solution_func = analytic_solution_func
    
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.loss(X, y, w)
        w_regularization = w.copy()
        w_regularization[-1] = 0.0
        penalty_part = 0.5 * self.mu_rate * float(w_regularization @ w_regularization)
        return core_part + penalty_part
    
    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.gradient(X, y, w)
        w_regularization = w.copy()
        w_regularization[-1] = 0.0
        penalty_part = self.mu_rate * w_regularization
        return core_part + penalty_part


class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \hat{y}
        """
        return X @ self.w

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        X: np.ndarray
        if X_batch is None:
            X = self.X_train
        else:
            X = X_batch
        y: np.ndarray
        if X_batch is None:
            y = self.y_train
        else:
            y = y_batch
        return self.loss_function.gradient(X, y, self.w)


    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        X: np.ndarray
        if X_batch is None:
            X = self.X_train
        else:
            X = X_batch
        y: np.ndarray
        if X_batch is None:
            y = self.y_train
        else:
            y = y_batch
        return self.loss_function.loss(X, y, self.w)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.
        
        X: np.ndarray, 
        y: np.ndarray
        """
        self.X_train, self.y_train = X, np.asarray(y)
        features_number = X.shape[1]
        self.w = np.zeros(features_number)
        self.optimizer.optimize()
        
class LogCosh(LossFunction):
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        r = X @ w - y
        return float(np.mean(np.log(np.cosh(r))))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        r = X @ w - y
        return (X.T @ np.tanh(r)) / r.shape[0]


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        r = X @ w - y
        ln = self.delta * np.abs(r) - 0.5 * self.delta**2
        return float(np.mean(np.where(np.abs(r) < self.delta, 0.5 * r**2, ln)))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        r = X @ w - y
        g = np.where(np.abs(r) < self.delta, r, self.delta * np.sign(r))
        return (X.T @ g) / r.shape[0]
    