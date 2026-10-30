"""비교 기준 1~3 (B팀 문서 '비교 기준'). 4번 MLP는 model.py (Transformer와 같은 학습 코드 사용).

1 mean      포지션 평균 위치: 상황과 상관없이 항상 같은 자리
2 ballshift 평균 + 공 따라 이동: 포지션마다 (공 x, 공 y, 공격/수비) 선형 회귀로 평균 자리에서 얼마나 움직이는지
3 knn       비슷한 장면 찾기: 공·국면·상대 배치가 가장 비슷한 학습 장면 k개에서 '평균 자리 대비 이동량' 평균
모두 fit(train) -> predict(D) -> (N, 10, 2) m.
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from data import SCALE, flat_features


class Mean:
    name = 'mean'

    def fit(self, D):
        return self

    def predict(self, D):
        return D['own_mean'].copy()


class BallShift:
    name = 'ballshift'

    @staticmethod
    def _X(D):
        b = D['ball'] / SCALE; ph = D['phase'][:, None].astype(float)
        return np.concatenate([b, ph, ph * b, np.abs(b[:, 1:2]), np.ones((len(b), 1))], 1)

    def fit(self, D):
        X = self._X(D); Y = (D['own'] - D['own_mean']).reshape(len(X), -1)
        self.W = np.linalg.lstsq(X, Y, rcond=None)[0]
        return self

    def predict(self, D):
        X = self._X(D)
        return D['own_mean'] + (X @ self.W).reshape(len(X), 10, 2)


class KNN:
    name = 'knn'

    def __init__(self, k=20, max_train=60000, seed=0):
        self.k, self.max_train, self.seed = k, max_train, seed

    W_BALL, W_PHASE = 3.0, 3.0                  # after standardising: ball and phase count more than one opponent

    @staticmethod
    def _X(D):
        return flat_features(D)[:, :23]          # ball, phase, opponents (our own role means are not a 'situation')

    def _scaled(self, X):
        Z = self.sc.transform(X)
        Z[:, :2] *= self.W_BALL; Z[:, 2] *= self.W_PHASE
        return Z

    def fit(self, D):
        X = self._X(D); Y = (D['own'] - D['own_mean']).reshape(len(X), -1)
        if len(X) > self.max_train:
            i = np.random.default_rng(self.seed).choice(len(X), self.max_train, replace=False); X, Y = X[i], Y[i]
        self.sc = StandardScaler().fit(X)
        # n_jobs=4: shared lab server, do not take every core
        self.m = KNeighborsRegressor(self.k, weights='distance', n_jobs=4).fit(self._scaled(X), Y)
        return self

    def predict(self, D):
        X = self._scaled(self._X(D))
        return D['own_mean'] + self.m.predict(X).reshape(len(X), 10, 2)
