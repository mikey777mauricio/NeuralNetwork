from LinearLayer import *
from ReLU import *

class Network:
    def __init__(self, layers_arch):
        '''
        /*  TODO: 1) Initialize the array for input layers with the proper feature sizes specified in the input vector.
         * For the linear layer, in each pair (in_size, out_size), the in_size is the feature size of the previous layer and the out_size is the feature size of the output (that goes to the next layer)
         * In the linear layer, the weight should have the shape (in_size, out_size).

         *  For example, if layers_arch = [['Linear', (256, 128)], ['ReLU'], ['Linear', (128, 64)], ['ReLU'], ['Linear', (64, 32)]],
       * 							 then there are three linear layers whose weights are with shapes (256, 128), (128, 64), (64, 32),
       * 							 and there are two non-linear layers.
         *  Attention: * The output feature size of the linear layer i should always equal to the input feature size of the linear layer i+1.
       */
        '''

        ########## Code start  ##########
        self.layers = []
        for i in range(0, len(layers_arch)):
            if layers_arch[i][0] == 'Linear':
                self.layers.append(LinearLayer(layers_arch[i][1][0], layers_arch[i][1][1]))
            else:
                self.layers.append(ReLU())

        ##########  Code end   ##########

    def forward(self, X):
        '''
        /*
         * TODO: propagate the input data for the first linear layer throught all the layers in the network and return the output of the last linear layer.
         * For implementation, you need to write a for-loop to propagate the input from the first layer to the last layer (before the loss function) by going through the forward functions of all the layers.
         * For example, for a network with k linear layers and k-1 activation layers, the data flow is:
         * linear[0] -> activation[0] -> linear[1] ->activation[1] -> ... -> linear[k-2] -> activation[k-2] -> linear[k-1]
         */
        '''

        ########## Code start  ##########
        output_layer = self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
            output_layer = self.layers[i].forward(output_layer)

        return output_layer

        ##########  Code end   ##########

    def backward(self, Y_grad):
        '''
        /* Propagate the gradient from the last layer to the first layer by going through the backward functions of all the layers.
         * TODO: propagate the gradient of the output (we got from the Forward method) back throught the network and return the gradient of the first layer.

         * Notice: We should use the chain rule for the backward.
         * Notice: The order is opposite to the forward.
         */
        '''

        ########## Code start  ##########
        output_layer = self.layers[-1].backward(Y_grad)
        for i in range(len(self.layers) - 2, -1, -1):
            output_layer = self.layers[i].backward(output_layer)

        return output_layer

        ##########  Code end   ##########