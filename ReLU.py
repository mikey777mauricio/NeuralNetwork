import math
import numpy as np

class ReLU:
    # sigmoid layer
    def __init__(self):
        self.stored_X = None  # Here we should store the input matrix X for Backward

    def forward(self, X):
        '''
        /*
         *  The input X matrix has the dimension [#samples, #features].
         *  The output Y matrix has the same dimension as the input X.
         *  You need to perform ReLU on each element of the input matrix to calculate the output matrix.
         *  TODO: 1) Create an output matrix by going through each element in input and calculate relu=max(0,x) and
         *  TODO: 2) Store the input X in self.stored_X for Backward.
         */
        '''

        ########## Code start  ##########
        output_Y = np.zeros([X.shape[0], X.shape[1]])
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                output_Y[i, j] = max(0, X[i, j])

        self.stored_X = X

        return output_Y

        ##########  Code end   ##########

    def backward(self, Y_grad):
        '''
         /*  grad_relu(x)=1 if relu(x)=x
         *  grad_relu(x)=0 if relu(x)=0
         *
         *  The input matrix has the name "Y_grad." The name is confusing (it is actually the input of the function). But the name follows the convension in PyTorch.
         *  The output matrix has the same dimension as input.
         *  The output matrix is calculated as grad_relu(stored_X)*Y_grad.
         *  TODO: returns the output matrix calculated above
         */
        '''

        ########## Code start  ##########
        grad_output = np.zeros([self.stored_X.shape[0], self.stored_X.shape[1]])
        for i in range(0, grad_output.shape[0]):
            for j in range(0, grad_output.shape[1]):
                if self.stored_X[i, j] < 0:
                    grad_output[i, j] = 0
                else:
                    grad_output[i, j] = 1

        return grad_output * Y_grad

        ##########  Code end   ##########