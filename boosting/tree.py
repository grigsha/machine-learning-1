import numpy as np


def find_best_split(feature_vector, grad_vector, hess_vector, l2=1.0):
    feature_vector = np.array(feature_vector)
    grad_vector = np.array(grad_vector)
    hess_vector = np.array(hess_vector)

    sorting_ord = np.argsort(feature_vector)
    sorted_features = feature_vector[sorting_ord]
    sorted_grad = grad_vector[sorting_ord]
    sorted_hess = hess_vector[sorting_ord]

    unique_vals = np.unique(sorted_features)
    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
    if len(thresholds) == 0:
        return thresholds, np.array([]), None, None
    left_tree = np.searchsorted(sorted_features, thresholds, side='right')
    grad_sum = np.cumsum(sorted_grad)
    hess_sum = np.cumsum(sorted_hess)

    grad_left = grad_sum[left_tree - 1]
    hess_left = hess_sum[left_tree - 1]
    grad_right = hess_sum[-1] - grad_left
    hess_right = grad_sum[-1] - hess_left
    score = (
        grad_left ** 2 / (hess_left + l2)
        + grad_right ** 2 / (hess_right + l2)
        - grad_sum[-1] ** 2 / (hess_sum[-1] + l2)
    )
    threshold_best = thresholds[score == np.max(score)][0]
    return thresholds, score, threshold_best, np.max(score)


class HessianTree:
    def __init__(self, max_depth=None, min_samples_split=None, min_samples_leaf=None, l2=1.0):
        self._tree = {}
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self.l2 = l2

    def _fit_node(self, sub_X, sub_grad, sub_hess, node, depth):
        if self._max_depth is not None and self._max_depth <= depth:
            node["type"] = "terminal"
            node["value"] = np.sum(sub_grad) / (np.sum(sub_hess) + self.l2)
            return

        if self._min_samples_split is not None and len(sub_grad) < self._min_samples_split:
            node["type"] = "terminal"
            node["value"] = np.sum(sub_grad) / (np.sum(sub_hess) + self.l2)
            return

        feature_best, threshold_best, score_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_vector = sub_X[:, feature]

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, score = find_best_split(feature_vector, sub_grad, sub_hess, self.l2)

            split_check = feature_vector < threshold
            if self._min_samples_leaf is not None:
                if np.count_nonzero(split_check) < self._min_samples_leaf:
                    continue
                if np.count_nonzero(~split_check) < self._min_samples_leaf:
                    continue

            if score_best is None or score > score_best:
                feature_best = feature
                score_best = score
                split = split_check
                threshold_best = threshold

        if feature_best is None:
            node["type"] = "terminal"
            node["value"] = np.sum(sub_grad) / (np.sum(sub_hess) + self.l2)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_grad[split], sub_hess[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_grad[~split], sub_hess[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["value"]
        feature_ind = node["feature_split"]
        if x[feature_ind] < node["threshold"]:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, grad, hess):
        X = np.asarray(X)
        grad = np.asarray(grad)
        hess = np.asarray(hess)
        self._fit_node(X, grad, hess, self._tree, 0)
        return self

    def predict(self, X):
        X = np.asarray(X)
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
