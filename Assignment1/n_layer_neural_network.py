#__author__ = Alberto Fung
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid min, y_max, h))
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


class DeepNeuralNetwork(NeuralNetwork):
    """
    This class inherits from NeuralNetwork and builds a DeepNetwork Class
    """

    def __init__(self, nn_input_dim, nn_hlayer_num, nn_hlayer_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hlayer_num: number of hidden layers
        :param nn_hlayer_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''

        self.nn_input_dim = nn_input_dim
        self.nn_hlayer_num = nn_hlayer_num
        self.nn_hlayer_dim = nn_hlayer_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.hLayer = Layer(self.nn_hlayer_num, self.actFun_type)

        # initialize the weights and biases in the network
        np.random.seed(seed)

        # initializes weights and biases for Output layer (L) and input layer(1)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hlayer_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hlayer_dim))
        self.WL = np.random.randn(self.nn_hlayer_dim, self.nn_output_dim) / np.sqrt(self.nn_hlayer_dim)
        self.bL = np.zeros((1, self.nn_output_dim))

        # initializes a N-dimension array to store each W and B for the hidden layers
        self.Wh = np.random.randn(self.nn_hlayer_dim, self.nn_hlayer_dim, nn_hlayer_num -1) / \
                           np.sqrt(self.nn_hlayer_dim)
        self.bh = np.zeros((1, self.nn_hlayer_dim, nn_hlayer_num -1))



    def actFun(self, z: object, type: object) -> object:
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if type == 'tanh':
            return np.tanh(z)
        if type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        if type == 'relu':
            return z * (z > 0)

        return None

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if type == 'tanh':
            return 1.0 - np.tanh(z) ** 2
        if type == 'sigmoid':
            return (1.0 / (1.0 + np.exp(-z))) * (1.0 - (1.0 / (1.0 + np.exp(-z))))
        if type == "relu":
            return 1.0 * (z > 0)

        return None

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = actFun(self.z1)
        self.zh = np.zeros((len(X), self.nn_hlayer_dim, self.nn_hlayer_num-1))
        self.ah = np.zeros((len(X), self.nn_hlayer_dim, self.nn_hlayer_num-1))
        #print(self.zh.shape)
        #print(self.ah.shape)

        self.zh[:, :, 0], self.ah[:, :, 0] = self.hLayer.feedforward(self.a1, self.Wh[:, :, 0], self.bh[:, :, 0], 0)
        #print(self.zh[:,:,0].shape)
        #print(self.ah[:,:,0].shape)

        #print(self.nn_hlayer_num-1)
        for i in range(1, self.nn_hlayer_num-1):
            #print("ran")
            self.zh[:, :, i], self.ah[:, :, i] = self.hLayer.feedforward(self.ah[:, :, i-1], self.Wh[:, :, i], self.bh[:, :, i], i)
            #print(np.shape(self.Wh[:,:,1]))
        self.zL = self.ah[:, :, -1].dot(self.WL) + self.bL

        exp_scores = np.exp(self.zL)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #print("yaya")
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)

        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))

        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss (optional)

        WSS = 0
        for i in range(0, self.nn_hlayer_num-1):
                WSS += np.sum(np.square(self.Wh[:, :, i]))
        WSS += np.sum(np.square(self.W1)) + np.sum(np.square(self.WL))
        data_loss += self.reg_lambda / 2 * WSS
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        deltaL = self.probs
        # print(delta3)
        deltaL[range(num_examples), y] -= 1
        deltaL /= num_examples

        dWL = self.a1.T.dot(deltaL)
        dbL = np.sum(deltaL, axis=0)
        #print(dbL.shape)

        dWh = np.zeros((self.nn_hlayer_dim, self.nn_hlayer_dim, self.nn_hlayer_num-1))
        dbh = np.zeros((1, self.nn_hlayer_dim, self.nn_hlayer_num-1))
        delta = np.zeros((len(X), self.nn_hlayer_dim, self.nn_hlayer_num-1))
        #print(delta.shape)
        #print(dWn.shape)
        #print(dbn.shape)


        dWh[:, :, -1], dbh[:, :, -1], delta[:, :, -1] = \
            self.hLayer.backprop(self.WL, self.zh[:, :, -1], self.ah[:, :, -1], deltaL)
        #print(dWn[:,:,-1])
        #print(dWn[:,:,3])


        for i in range(2, self.nn_hlayer_num):
            dWh[:, :, -i], dbh[:, :, -i], delta[:, :, -i] = \
                self.hLayer.backprop(self.Wh[:, :, -i+1], self.zh[:, :, -i], self.ah[:, :, -i], delta[:, :, -i+1])
            #print(i)

        #for i in range(0, self.nn_hlayer_num-1):
           # print(i)
           # print(dWn[:, :, i])

        dW1 = X.T.dot(delta[:,:,0])
        db1 = np.sum(delta[:,:,0])


        return dW1, dWL, db1, dbL, dWh, dbh

    def fit_model(self, X, y, epsilon=0.001, num_passes=10000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.

        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW1, dWL, db1, dbL, dWh, dbh = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            # print(self.reg_lambda)
            dWL += self.reg_lambda * self.WL
            dW1 += self.reg_lambda * self.W1
            dWh += self.reg_lambda * self.Wh

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.WL += -epsilon * dWL
            self.bL += -epsilon * dbL
            self.Wh += -epsilon * dWh
            self.bh += -epsilon * dbh

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)

class Layer(NeuralNetwork):

    def __init__(self, nn_hlayer_num, actFun_type='tanh'):
        '''

        :param nn_hlayer_num: number of hidden layers
        :param actFun_type: activation function type
        '''

        self.nn_layer_num = nn_hlayer_num
        self.actFun_type = actFun_type



    def feedforward(self, X, W, b, i):
        '''
        Makes a feedforward step for a single layer
        :param X: neuron input
        :param W: Weight of input layer
        :param b: biases of input layer
        :param actFun: activation function type
        :return: returns z and f(z)
        '''

        #print("X", i, X.shape)
        #print("W",i ,W.shape)
        #print("b", i ,b.shape)
        z = X.dot(W) + b
        a = self.actFun(z, self.actFun_type)
        #a = self.actFun_type(z, self.actFun_type)
        return z, a

    def backprop(self, W, z, a, delta):
        '''
        :param W: Weight of previous layer
        :param z: sum of previous layer
        :param deltaprev: delta of previous layer
        :param actFun: activation funtion type
        :return: deltanext
        '''

        deltaprev = delta.dot(W.T) * self.diff_actFun(z, type=self.actFun_type)
        dW = a.T.dot(deltaprev)
        db = np.sum(deltaprev, axis=0)

        return dW, db, deltaprev

def main():
    # # generate and visualize Make-Moons dataset

    X, y = generate_data()
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)


    plt.title("Make-Moon Dataset")
    #plt.show()


    #Initializes the NN with parameters
    model = DeepNeuralNetwork(nn_input_dim=2, nn_hlayer_num=6, nn_hlayer_dim=6, nn_output_dim=2, actFun_type='tanh', reg_lambda=0.01, seed=0)




    #Train on the dataset with the created NN
    model.fit_model(X,y)


    model.visualize_decision_boundary(X,y)


if __name__ == "__main__":
    main()