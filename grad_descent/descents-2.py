import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """

    def __init__(self,
                 lr_schedule: LearningRateSchedule = TimeDecayLR(),
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                 ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        self.iteration = 0
        loss_hist = [self.model.compute_loss()]
        for i in range(self.max_iter):
            d = self._step()
            loss_hist.append(self.model.compute_loss())
            if d @ d < self.tolerance:
                break
            if np.isnan(d).any():
                break
        self.model.loss_history = loss_hist


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        etha_k = self.lr_schedule.get_lr(self.iteration)
        w_before = self.model.w.copy()
        grad = self.model.compute_gradients(X_train, y_train)
        self.model.w -= etha_k * grad
        return self.model.w - w_before


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        object_cnt = X_train.shape[0]
        indexes = np.random.randint(0, object_cnt, self.batch_size)
        etha_k = self.lr_schedule.get_lr(self.iteration)
        w_before = self.model.w.copy()
        grad = self.model.compute_gradients(X_train[indexes], y_train[indexes])
        self.model.w -= etha_k * grad
        return self.model.w - w_before


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, warmup_memory_iterations=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.avg_grad = None
        self.batch_size = batch_size
        self.warmup_memory_iterations = warmup_memory_iterations

    def _update_memory(self):
        object_cnt, features_cnt = self.model.X_train.shape
        indexes = np.random.randint(0, object_cnt, self.batch_size)
        grad = np.vstack([self.model.compute_gradients(self.model.X_train[j:j + 1], self.model.y_train[j:j + 1])
                          for j in indexes])
        self.avg_grad += (grad - self.grad_memory[indexes]).sum(axis=0) / object_cnt
        self.grad_memory[indexes] = grad

    def _update_weights(self) -> np.ndarray:
        object_cnt, features_cnt = self.model.X_train.shape
        if self.grad_memory is None:
            self.grad_memory = np.zeros((object_cnt, features_cnt))
            self.avg_grad = np.zeros((features_cnt))
            for warmup_iteration in range(self.warmup_memory_iterations):
                self._update_memory()

        etha_k = self.lr_schedule.get_lr(self.iteration)
        w_before = self.model.w.copy()
        self._update_memory()
        self.model.w -= etha_k * self.avg_grad
        return self.model.w - w_before


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        features_cnt = X_train.shape[1]
        if self.velocity is None:
            self.velocity = np.zeros((features_cnt))
        etha_k = self.lr_schedule.get_lr(self.iteration)
        w_before = self.model.w.copy()
        grad = self.model.compute_gradients(X_train, y_train)
        self.velocity = self.beta * self.velocity + etha_k * grad
        self.model.w -= self.velocity
        return self.model.w - w_before


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        features_cnt = X_train.shape[1]
        if self.m is None:
            self.m = np.zeros((features_cnt))
        if self.v is None:
            self.v = np.zeros((features_cnt))
        etha_k = self.lr_schedule.get_lr(self.iteration)
        w_before = self.model.w.copy()
        grad = self.model.compute_gradients(X_train, y_train)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        m_k = self.m / (1 - self.beta1 ** (self.iteration + 1))
        v_k = self.v / (1 - self.beta2 ** (self.iteration + 1))
        self.model.w -= (etha_k * m_k) / (v_k ** 0.5 + self.eps)
        return self.model.w - w_before


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """

    def __init__(self):
        self.model = None

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        X = self.model.X_train
        y = self.model.y_train
        loss_func = self.model.loss_function
        self.model.w = loss_func.analytic_solution(X, y)
        self.model.loss_history = [self.model.compute_loss(X, y)]
