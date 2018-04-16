import numpy as np


class DNN:

    def __init__(self, shape=[]):
        '''
        Sets the shape of the dense neural net.
        Example: if the neural net has 3 input nodes, hidden layers with the first having
        3 nodes and the second having 2 nodes, with an output of 2 nodes, then we would
        have shape = [3,3,2,2]
        '''
        self.layers = []
        self.weights = []
        self.biases = []
        
        for layer in shape:
            self.layers.append(np.zeros(shape=(layer)))
            self.weights.append(0)
            self.biases.append(0)
            
        for l in range(1,len(shape)):
            self.weights[l] = np.random.random_sample(size=(shape[l],shape[l-1]))
            self.biases[l] = np.random.random_sample(size=(shape[l]))      

    def setLayerWeights(self, layer, weight):
        '''
        Sets the weight array for the layer specified. 
        Weights for layer i correspond to the transition from layer i-1 to layer i.
        
        :param layer: the layer number as an integer
        :param weight: the weight numpy array to be assigned
        '''
        if len(weight) == len(self.weights[layer]):
            self.weights[layer] = weight
        else:
            raise Exception('Weight array incorrect size')
            
    def setLayerBiases(self, layer, bias):
        '''
        Sets the bias array for the layer specified. 
        
        :param layer: the layer number as an integer
        :param bias: the bias numpy array to be assigned
        '''
        if len(bias) == len(self.biases[layer]):
            self.biases[layer] = bias
        else:
            raise Exception('Bias array incorrect size')
        
    def setActivationFunction(self, activation, activationPrime):
        '''
        Sets the activation function to be used.
        The derivative of the activation function must also be entered
        in order to optimize the weights and balances.
        
        :param activation: the activation function to be used
        :param activationPrime: the derivative of the activation function
        '''
        self.activation = activation
        self.activationPrime = activationPrime
        
    def setCost(self, cost, costPrime):
        '''
        Sets the cost function to be used.
        The derivative of the cost function must also be entered in order
        to optimize the weights and balances.
        Both the cost and costPrime functions are assumed to be a function
        of two variables, the actual and estimated y-values. i.e cost(actual,estimated)
        
        :param cost: the cost function to be used
        :param costPrime: the derivative of the cost function
        '''
        self.cost = cost
        self.costPrime = costPrime
        
    def propogate(self,input):
        '''
        Propogates the input through the net.
        
        :param input: a numpy array to be input and propogated forward through the net.
            Each row corresponds to one training example
        '''
        self.layers[0] = input
        for i in range(1,len(self.layers)):
            self.layers[i] =  self.activation(np.matmul(self.layers[i-1], np.transpose(self.weights[i])) + self.biases[i])