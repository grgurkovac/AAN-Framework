import numpy as np
import math
import random
import pickle


class FeedForwardNet:

    @staticmethod
    def sigmoid(xs, derivate=False):
        """
        applies a sigmoid function or its derivative to each element of the given vector
        :param xs: vector to apply the function to
        :param derivate: if True then apply a derivative of the function, otherwise apply the function
        :return: a vector with the applied function
        """
        if not derivate:
            func = np.vectorize(lambda x: 1.0 / (1 + math.exp(-x)))
            return func(xs)
        else:
            funcd = np.vectorize(lambda x: np.exp(x) / (np.exp(x) + 1) ** 2)
            return funcd(xs)

    # f(x)=x f'(x)=1
    @staticmethod
    def nothing(xs, derivate=False):
        """
        applies a function f(x)=x or its derivative to each element of the given vector
        :param xs: vector to apply the function to
        :param derivate: if True then apply a derivative of the function, otherwise apply the function
        :return: a vector with the applied function
        """
        if not derivate:
            return xs
        else:
            funcd = np.vectorize(lambda x: 1)
            return funcd(xs)


    @staticmethod
    def relu(xs, derivate=False):
        """
        applies a ReLU function (rectified linear unit, f(x)=max(x,0)) or its derivative to each element of the given vector
        :param xs: vector to apply the function to
        :param derivate: if True then apply a derivative of the function, otherwise apply the function
        :return: a vector with the applied function
        """
        if not derivate:
            func = np.vectorize(lambda x: x if x >= 0 else 0)
            return func(xs)
        else:
            funcd = np.vectorize(lambda x: 1 if x >= 0 else 0)
            return funcd(xs)

    @staticmethod
    def leaky_relu(xs, derivate=False):
        """
        applies a leaky ReLU function (leaky rectified linear unit) or its derivative to each element of the given vector
        leaky ReLU is defined as follows
            f(x) = x , x >= 0
            f(x) = 0.01*x , x < 0
        :param xs: vector to apply the function to
        :param derivate: if True then apply a derivative of the function, otherwise apply the function
        :return: a vector with the applied function
        """
        if not derivate:
            func = np.vectorize(lambda x: x if x >= 0 else 0.01*x)
            return func(xs)
        else:
            funcd = np.vectorize(lambda x: 1 if x >= 0 else 0.01)
            return funcd(xs)

    @staticmethod
    def tanh(xs, derivate=False):
        """
        applies a hyperbolic tangent function (tanh) or its derivative to each element of the given vector
        :param xs: vector to apply the function to
        :param derivate: if True then apply a derivative of the function, otherwise apply the function
        :return: a vector with the applied function
        """
        if not derivate:
            func = np.vectorize(math.tanh)
            return func(xs)
        else:
            funcd = np.vectorize(lambda x: 1 - math.tanh(x) ** 2)
            return funcd(xs)

    @staticmethod
    def error_partial_sum_squared(os, ys):
        """
        A partial derivative of the sum squared error function, with respect to the input to the neuron.
        y-o, where y is the output, and o is the expected output
        :param ys: np.matrix containing outputs
        :param os: np.matrix containing expected outputs
        :return: the partial derivative matrixl
        """
        return np.subtract(os, ys)

    error_partial_dict = {
        "error_partial_sum_squared": error_partial_sum_squared
    }

    def __init__(self,
                 input_count=None, layers=None,
                 activation_function=None, functions=None,
                 weights=None, weights_initializer=None, error_partial=None):
        """
        Creates a new FeedForwardNet object.
        
        :param input_count: A number of inputs. 
        :param layers: An array of integers representing a number of neurons in each layer of the net.
        :param activation_function: 
            An activation function to be used in every neuron in the net.
            If this is not None, parameter functions should be.
            If both activation_function and functions is None defaults to FeedForwardNet.sigmoid
        :param functions: 
            A list of functions, representing a function to be used in each layer of the net.
            If this is None parameter activation_function is used
            If this is not None, parameter activation_function should be. 
            If activation_function is not None it is ignored by functions.
        :param weights: 
            A list of 2D numpy arrays representing weight for each layer.
            If weights is None. A weights_initializer is used. It is recommended to use an initializer.
        :param weights_initializer: 
            A function that initializes starting weights. 
            If None defaults to self.xavier_weight_initializer.
        :param error_partial:
            Defines a error function partial to be minimized. 
            If this is None defaults to FeedForwardNet.error_partial_sum_squared.
        """
        self.input_count = input_count
        self.layers = layers
        self.layers_count = len(self.layers)
        if functions is not None:
            self.functions = functions
        elif activation_function is not None:
            self.functions = [activation_function] * len(layers)
        else:
            self.functions = [FeedForwardNet.sigmoid] * len(layers)

        if weights_initializer is None:
            # defaults to xavier weights initalizer
            self.weights_initializer=self.xavier_weight_initializer

        # Error function
        if error_partial is None:
            self.error_partial = FeedForwardNet.error_partial_sum_squared
        else:
            self.error_partial = error_partial

        if weights is None:
            self.weights_initializer()
        else:
            self.weights = weights


        self.X = [None] * self.layers_count  # represents input
        self.IN = [None] * self.layers_count  # represents input after we sum it and apply the weights
        self.Y = [None] * self.layers_count  # represents output of the activation function


    def xavier_weight_initializer(self):
        """
        Initializes the weights using the xavier initialization.
        Uses numpy.random.normal with 
        mean=0 and standard deviation = to 1/sqrt(number of weights preceding a neuron)
        Saves the weights in self.weights
        """
        self.weights=[None] * self.layers_count
        self.weights[0] = np.random.normal(
            loc=0,
            scale=1 / math.sqrt(self.input_count + 1),
            size=(self.input_count + 1, self.layers[0]))
        for i in range(1, len(self.layers)):
            self.weights[i] = np.random.normal(
                loc=0,
                scale=1 / math.sqrt(self.layers[i - 1] + 1),
                size=(self.layers[i - 1] + 1, self.layers[i]))


    def random_weight_initializer(self):
        """
        Initializes the weights using a basic numpy.random.rand function.
        Saves the weights in self.weights
        """
        self.weights = [np.random.rand(self.input_count + 1, self.layers[0])]  # plus one for biases
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.rand(self.layers[i - 1] + 1, self.layers[i]))  # plus one for biases





    def save_state(self, filename):
        """
        Saves the net configuration using pickle
        :param filename: name of the file
        """
        with open(filename,"wb") as file:
            pickle.dump(self, file)
        return


    @staticmethod
    def load_state(filename):
        """
        Loads the net configuration from a file using pickle
        :param filename: name of the file
        """
        with open(filename,"rb") as file:
            net = pickle.load(file)
        return net

    def check_maximum_error(self, output_values, epsilon , Y=None , verbose=False, error_list=None):
        """
        Calculates the maximum of all the differences between the output of every neuron on every example and its corresponding expected output.
        Returns true if that error is smaller or equal to epsilon
        :param output_values: expected output
        :param epsilon: decimal number representing the error limit
        :param Y: output, if not given it is filled with self.Y[-1] (values from the last forward_propagation)
        :param verbose: if true maximum error will be printed to standard output
        ;:param error_list: maximum error value will be added to the end of the list
        :return: True if |o - y| <= epsilon for each y, and its corresponding o, False otherwise
        """
        if Y == None:
            Y=self.Y[-1]
        maximum_error=np.amax(np.abs(np.subtract(output_values,Y)))
        if verbose: print("maximum error:", maximum_error)
        if error_list is not None: error_list.append(maximum_error)
        return maximum_error <= epsilon

    def check_total_squared_error(self, output_values, epsilon, Y=None, verbose=False, error_list=None):
        """
        Calculates the sum of all the squared differences between the output of every neuron in every example and its corresponding expected output
        Returns true if that error is smaller or equal to epsilon
        :param output_values: expected output
        :param epsilon: decimal number representing the error limit
        :param Y: output, if not given it is filled with self.Y[-1] (values from the last forward_propagation)
        :param verbose: If true total squared error will be printed to standard output
        :param error_list: total squared error value will be added to the end of the list
        :return: True if (o1 - y1)^2 + .... + (on - y1)^2 <= epsilon
        """
        if Y == None:
            Y=self.Y[-1]
        total_squared_error = np.sum(np.square(np.subtract(output_values,Y)))
        if verbose: print("total squared error:",total_squared_error)
        if error_list is not None: error_list.append(total_squared_error)
        return total_squared_error <= epsilon

    def generate_random_batch(self, input_values, output_values, batch_size):
        """
        Takes a random batch_size of examples from input_values, and output_values
        :param input_values: an matrix of input examples
        :param output_values: an matrix of their corresponding outputs
        :param batch_size: a number of examples to be selected
        :return: a matrix containing selected examples's input_values, 
        and a matrix containing their corresponding output values
        """
        if input_values.shape[0] != output_values.shape[0]:
            raise ValueError("input_values and output_values must represent the same number of examples")
        examples_count = input_values.shape[0]
        if batch_size > examples_count:
            raise ValueError(
                "batch_size must be less or equal to number of examples represented by input_values and output_values. "
                "\nbatch_size: "+str(batch_size)+" examples_count: "+str(examples_count)
            )

        #randomizes the indexes
        indexes=random.sample(range(examples_count),batch_size)
        return input_values[indexes],output_values[indexes]


    def forward_propagation(self, input_values):
        """
        Propagates the inputs and caluclates the X, IN, and Y variables that are later used during backpropagation
        :param input_values: a 2D array representing inputs of multiple iterations,
        for example:
            input_values[0] represents the input values of the first iteration
            input_values[3][0] represents the first input of the fourth iteration
        :return: returns the output of all iterations 
        """
        batch_size = input_values.shape[0]  # pythonic? da ovo primam preko argumenata??
        self.IN[0] = np.append(input_values, [[-1]] * batch_size, axis=1)
        for i in range(0, self.layers_count):
            self.X[i] = (np.matmul(self.IN[i], self.weights[i]))
            self.Y[i] = (self.functions[i](self.X[i]))
            if i != self.layers_count - 1:
                # -1 represents a bias, we add it to each iteration
                self.IN[i + 1] = np.append(self.Y[i], [[-1]] * batch_size, axis=1)
        return self.Y[-1]

    def backpropagation(self, output_values, learning_rate, inertion_factor=None):
        """
        Updates the weights using backpropacation, error is sumed over all traning examples.
        Note that this method updates the weights only once.
        Also note that forward_propagation must be run before using this method.
        forward_propagation argument input_values must be of the same length as the output_values of this method
        :param output_values: 2D numpy array representing the expected outputs 
            output_values[i] represent expected outputs of the i-th iteration
            output_values[i][j] represents the output on the j-th neuron 
        :param learning_rate: a decimal number
        :inertion_factor: decimal number representing the part of the previos update to be carried on to the current update
        w = w + update, where update = non_inertial_update + inertion_factor * previous_update
        :return: returns the updated weights
        """

        if inertion_factor is not None and inertion_factor != 0:
            # inertion enabled
            intertion = True
            if not hasattr(self, 'former_update'):
                # the variable former_update does not exist
                # this is the first call of this function we need to initialize former_updates variable
                self.former_update = [None] * self.layers_count
        else:
            #intertion not enabled
            intertion = False

        deltas = [None] * self.layers_count
        deltas[-1] = np.multiply(
            self.error_partial(output_values, self.Y[-1]),
            self.functions[-1](self.X[-1], True)
        )

        for i in range(self.layers_count - 2, -1, -1):
            # deltas[i] i-th layer, deltas[i][j] j-th iteration, deltas[i][j][k] k-th neuron
            deltas[i] = np.multiply(
                np.matmul(self.weights[i+1][:-1], deltas[i + 1].transpose()).transpose(),
                self.functions[i](self.X[i], True)
            )




        for i in range(0, self.layers_count):
            update=np.multiply(
                    learning_rate,
                    np.matmul(
                        self.IN[i].transpose(),
                        deltas[i]
                    )
                )

            if intertion:
                #inertion enabled
                if self.former_update[i] is not None:
                    # this is not the first call of this function, therefore former updates exist
                    update = np.add(update, np.multiply(inertion_factor, self.former_update[i]))
                #save the current update for future updates
                self.former_update[i] = update

            self.weights[i] = np.add(
                self.weights[i],
                update
            )

        return self.weights

    def calculate_confusion_matrix(self, ground_truth_matrix, output_matrix=None):
        """
        Calculates a confusion matrix. A 2D matrix with the size of possible outputs x possible outputs.
        Where confusion_matrix[y][x] represents a number of times the net for an wanted output x predicted y.
        Biggest numbers should be on the diagonal.
        :param ground_truth_matrix: a matrix containing the wanted outputs
        :param output_matrix: a matrix containing the actual outputs, 
        if not given the last propagated values will be used ( self.Y[-1] )
        :return: numpy matrix representing the confusion matrix
        """
        if output_matrix is None:
            output_matrix=self.Y[-1]
        output=FeedForwardNet.one_hots_to_ints(output_matrix)
        ground_truth = FeedForwardNet.one_hots_to_ints(ground_truth_matrix)
        # confusion_matrix = np.zeros(shape=(self.layers[-1],self.layers[-1]))
        confusion_matrix = [[0]*self.layers[-1] for i in range(self.layers[-1])]
        number_of_examples=ground_truth.size

        for i in range(0,number_of_examples):
            confusion_matrix[output[i]][ground_truth[i]]+=1

        return np.matrix(confusion_matrix)

    @staticmethod
    def int_to_one_hot(num,min=0,max=1):
        """
        Converts a number to a one_hot vector.
        Creates a numpy array filled with min, and max on the num-th place
        :param num: index of the max
        :return: a numpy array
        """
        one_hot = np.zeros(10)
        one_hot.fill(min)
        one_hot[num] = max
        return one_hot

    @staticmethod
    def one_hots_to_ints(one_hots):
        """
        Converts a vector of vectors to a vector of numbers. Uses Net.one_hot_to_int function.
        :param one_hots: a 2D array
        :return: numpy array of ints, representing indices of maximum values in the corresponding vectors
        """
        rez = np.array([],dtype=np.int)
        for one_hot in one_hots:
            num = FeedForwardNet.one_hot_to_int(one_hot)
            rez = np.append(rez, num)
        return rez

    @staticmethod
    def one_hot_to_int(one_hot):
        """
        Converts a vector of floats to a number. In other words an argmax function.
        :param one_hot: a vector
        :return: the index containing the max value
        """
        max = one_hot[0]
        num = 0
        for i in range(1, 10):
            if one_hot[i] > max:
                max = one_hot[i]
                num = i
        return num