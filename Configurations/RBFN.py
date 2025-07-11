
# ------------------------------------------------------------------------

import numpy as np
from sklearn.cluster import KMeans

class RBFN(object):
    def __init__(self, input_shape, hidden_shape, kernel):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.sigma = None
        self.centers = None
        self.weights = None
        self.bias = None
        def Gaussianfun(center, data_point):  # Gaussian function
            return np.exp(-0.5 * np.power(np.linalg.norm(center - data_point) / self.sigma, 2))
        def Reflectedfun(center, data_point):  # Reflected function
            return 1/(1 + np.exp(np.power(np.linalg.norm(center - data_point) / self.sigma, 2)))
        def Multiquadric(center, data_point):  # Multiquadric function
            return np.sqrt(np.power(np.linalg.norm(center - data_point), 2) + np.power(self.sigma, 2))
        def INMultiquadric(center, data_point):  #  Inverse multiquadric function
            return 1/np.sqrt(np.power(np.linalg.norm(center - data_point), 2) + np.power(self.sigma, 2))
        if kernel == 'gaussian':
            self.kernel_ = Gaussianfun
        elif kernel == 'reflect':
            self.kernel_ = Reflectedfun
        elif kernel == 'mul':
            self.kernel_ = Multiquadric
        elif kernel == 'inmul':
            self.kernel_ = INMultiquadric

    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((X.shape[0], self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg,center_arg] = self.kernel_(center, data_point)
        return G

    def calsigma(self):
        max = 0.0
        num = 0
        total = 0.0
        for i in range(self.hidden_shape-1):
            for j in range(i+1,self.hidden_shape):
                dis = np.linalg.norm(self.centers[i] - self.centers[j])
                total = total + dis
                num += 1
                if dis > max:
                    max = dis
        self.sigma = 2*total/num

    def fit(self,X,Y):
        # if self.hidden_shape == int(X.shape[0]):
        #     self.centers = X
        # else:
        self.hidden_shape = int(X.shape[0] / 2)
        km = KMeans(n_clusters=self.hidden_shape).fit(X)
        self.centers = km.cluster_centers_
        self.calsigma()
        G = self._calculate_interpolation_matrix(X)
        temp = np.ones((len(X)))
        temp = np.column_stack((G, temp))
        temp = np.dot(np.linalg.pinv(temp), Y)
        self.weights = temp[:self.hidden_shape]
        self.bias = temp[self.hidden_shape]


    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights) + self.bias
        return predictions