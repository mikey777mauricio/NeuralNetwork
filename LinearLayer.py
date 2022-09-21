import math
import numpy as np


class LinearLayer:
    def __init__(self, _m, _n):
        '''
        :param _m: _m is the input X hidden size
        :param _n: _n is the output Y hidden size
        '''
        # "Kaiming initialization" is important for neural network to converge. The NN will not converge without it!
        self.W = (np.random.uniform(low=-900.0, high=900.0, size=(_m, _n))) / 100000.0 * np.sqrt(6.0 / _m)
        self.stored_X = None
        self.W_grad = None  # record the gradient of the weight

    def forward(self, X):
        '''
        :param X: shape(X)[0] is batch size and shape(X)[1] is the #features
         (1) Store the input X in stored_data for Backward.
         (2) :return: X * weights
        '''

        ########## Code start  ##########
        #         print(f'X: {X@self.W}')
        self.stored_X = X

        return X @ self.W
        ##########  Code end   ##########

    def backward(self, Y_grad):
        '''
        /* shape(output_grad)[0] is batch size and shape(output_grad)[1] is the # output features (shape(weight)[1])
         * 1) Calculate the gradient of the output (the result of the Forward method) w.r.t. the **W** and store the product of the gradient and Y_grad in W_grad
         * 2) Calculate the gradient of the output (the result of the Forward method) w.r.t. the **X** and return the product of the gradient and Y_grad
         */
        '''

        ########## Code start  ##########
        # gradient of output with respect to W
        self.W_grad = (self.stored_X).T @ Y_grad

        return Y_grad @ (self.W).T

        ##########  Code end   ##########