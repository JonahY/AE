from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, auc, confusion_matrix


class KernelKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="rbf", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)


def ICA(dim, *args):
    n = len(args)
    x = np.zeros([args[0].shape[0], n])
    for i in range(n):
        x[:, i] = args[i]

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_nor = (x - x_mean) / x_std

    ica = FastICA(n_components=dim)
    S_ = ica.fit_transform(x_nor)  # 重构信号
    A_ = ica.mixing_  # 获得估计混合后的矩阵
    return S_, A_


def plot_confmat(tn, fp, fn, tp):
    cm = np.zeros([2, 2])
    cm[0][0], cm[0][1], cm[1][0], cm[1][1] = tn, fp, fn, tp
    f, ax=plt.subplots(figsize=(2.5, 2))
    sns.heatmap(cm,annot=True, ax=ax, fmt='.20g') #画热力图
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(bottom=False,top=False,left=False,right=False)


def print_res(model, x_pred, y_true):
    target_pred = model.predict(x_pred)
    true = np.sum(target_pred == y_true)
    print('预测对的结果数目为：', true)
    print('预测错的的结果数目为：', y_true.shape[0]-true)
    print('使用SVM预测的准确率为：',
          accuracy_score(y_true, target_pred))
    print('使用SVM预测的精确率为：',
          precision_score(y_true, target_pred))
    print('使用SVM预测的召回率为：',
          recall_score(y_true, target_pred))
    print('使用SVM预测的F1值为：',
          f1_score(y_true, target_pred))
    print('使用SVM预测b的Cohen’s Kappa系数为：',
          cohen_kappa_score(y_true, target_pred))
    print('使用SVM预测的分类报告为：','\n',
          classification_report(y_true, target_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, target_pred).ravel()
    plot_confmat(tn, fp, fn, tp)
    return target_pred