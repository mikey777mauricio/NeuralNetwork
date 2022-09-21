import math
import numpy as np


class MSELoss:
    # cross entropy loss
    # return the mse loss mean(y_j-y_pred_i)^2

    def __init__(self):
        self.stored_diff = None

    def forward(self, prediction, groundtruth):
        '''
        /*  TODO: 1) Calculate stored_data=pred-truth
         *  TODO: 2) Calculate the MSE loss as the squared sum of all the elements in the stored_data divided by the number of elements, i.e., MSE(pred, truth) = ||pred-truth||^2 / N, with N as the total number of elements in the matrix
         */
        '''

        ########## Code start  ##########
        #         print(prediction)
        self.stored_diff = prediction - groundtruth
        return np.sum(np.square(self.stored_diff)) / (prediction.shape[0])

        ##########  Code end   ##########

    # return the gradient of the input data
    def backward(self):
        '''
        /* TODO: return the gradient matrix of the MSE loss
         * The output matrix has the same dimension as the stored_data (make sure you have stored the (pred-truth) in stored_data in your forward function!)
         * Each element (i,j) of the output matrix is calculated as grad(i,j)=2(pred(i,j)-truth(i,j))/N
         */
        '''

        ########## Code start  ##########
        output_grad = np.zeros([self.stored_diff.shape[0], self.stored_diff.shape[1]])
        output_grad = 2 * (self.stored_diff) / (output_grad.shape[0])
        return output_grad

        ##########  Code end   ##########