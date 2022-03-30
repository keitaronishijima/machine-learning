import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

class RegularizedLogisticRegression(object):
    '''
    Implement regularized logistic regression for binary classification.

    The weight vector w should be learned by minimizing the regularized loss
    \l(h, (x,y)) = log(1 + exp(-y <w, x>)) + \lambda \|w\|_2^2. In other words, the objective
    function that we are trying to minimize is the log loss for binary logistic regression 
    plus Tikhonov regularization with a coefficient of \lambda.
    '''
    def __init__(self):
        self.learningRate = 0.0001 # Feel free to play around with this if you'd like, though this value will do
        self.num_epochs = 1000 # Feel free to play around with this if you'd like, though this value will do
        self.batch_size = 10 # Feel free to play around with this if you'd like, though this value will do
        self.weights = 0

        #####################################################################
        #                                                                    #
        #    MAKE SURE TO SET THIS TO THE OPTIMAL LAMBDA BEFORE SUBMITTING    #
        #                                                                    #
        #####################################################################

        self.lmbda = 1 # tune this parameter
    def train(self, X, Y):
        '''
        Train the model, using batch stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None
        '''
        #[TODO]
        #print(X.shape)
        self.weights = np.zeros((1, X.shape[1]))
        epoch = 0

        num_examples = X.shape[0]
        while epoch < self.num_epochs:
            epoch += 1
            shuffler = np.random.permutation(len(Y))
            X = X[shuffler]
            Y = Y[shuffler]
            for i in range(num_examples//self.batch_size):
                X_batch = X[i * self.batch_size: (i+1) * self.batch_size]
                Y_batch = Y[i * self.batch_size: (i+1) * self.batch_size]
                delta_L = np.zeros((1, X.shape[1]))
                
                for example_no in range(self.batch_size):
                    #print(self.weights @ X_batch[example_no].T)
                    h = sigmoid_function(self.weights @ X_batch[example_no].T)[0]
                    delta_L += ((h - Y_batch[example_no]) * X_batch[example_no]/num_examples)
                delta_L +=  2 * self.lmbda * self.weights[0]
                self.weights = self.weights - (self.learningRate * delta_L / self.batch_size)
            loss = 0
            # for i in range(num_examples):
            #     h = sigmoid_function(self.weights @ X[i].T)[0]
            #     loss += Y[i] * np.log(h) + (1-Y[i]) * np.log(1 - h)
            # print(-loss/num_examples)
                

    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        predicted = sigmoid_function(X @ self.weights.T)
        res = np.zeros((predicted.shape[0],predicted.shape[1]), int)
        for i in range(predicted.shape[0]):
            if predicted[i] >= 0.5:
                res[i] = int(1)
            else:
                res[i] = int(0)
        return res.flatten()
        #[TODO]

    def accuracy(self,X, Y):
        '''
        Output the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        #[TODO]      
        predictions = self.predict(X)
        sum = 0
        #print("Predictions: ", predictions)
        #print("Y: ", Y)
        #print(Y) #train 398, , validation 85,
        for i in range(len(Y)):
            if predictions[i] == Y[i]:
                sum += 1
        return sum / Y.shape[0]

    def runTrainTestValSplit(self, lambda_list, X_train, Y_train, X_val, Y_val):
        '''
        Given the training and validation data, fit the model with training data and test it with
        respect to each lambda. Record the training error and validation error, which are equivalent 
        to (1 - accuracy).

        @params:
            lambda_list: a list of lambdas
            X_train: a 2D Numpy array for trainig where each row contains an example,
            padded by 1 column for the bias
            Y_train: a 1D Numpy array for training containing the corresponding labels for each example
            X_val: a 2D Numpy array for validation where each row contains an example,
            padded by 1 column for the bias
            Y_val: a 1D Numpy array for validation containing the corresponding labels for each example
        @returns:
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
        '''
        train_errors = []
        val_errors = []
        #[TODO] train model and calculate train and validation errors here for each lambda
        for l in lambda_list:
            self.lmbda = l
            self.train(X_train, Y_train)
            train_errors.append(1-self.accuracy(X_train, Y_train))
            val_errors.append(1-self.accuracy(X_val, Y_val))
        print("train errors: ", train_errors)
        print("val errors: ", val_errors)
        return train_errors, val_errors

    def _kFoldSplitIndices(self, dataset, k):
        '''
        Helper function for k-fold cross validation. Evenly split the indices of a
        dataset into k groups.

        For example, indices = [0, 1, 2, 3] with k = 2 may have an output
        indices_split = [[1, 3], [2, 0]].
        
        Please don't change this.
        @params:
            dataset: a Numpy array where each row contains an example
            k: an integer, which is the number of folds
        @return:
            indices_split: a list containing k groups of indices
        '''
        num_data = dataset.shape[0]
        fold_size = int(num_data / k)
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        indices_split = np.split(indices[:fold_size*k], k)
        return indices_split

    def runKFold(self, lambda_list, X, Y, k = 3):
        '''
        Run k-fold cross validation on X and Y with respect to each lambda. Return all k-fold
        errors.
        
        Each run of k-fold involves k iterations. For an arbitrary iteration i, the i-th fold is
        used as testing data while the rest k-1 folds are combined as one set of training data. The k results are
        averaged as the cross validation error.

        @params:
            lambda_list: a list of lambdas
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
            k: an integer, which is the number of folds, k is 3 by default
        @return:
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        '''
        k_fold_errors = []
        for lmbda in lambda_list:
            self.lmbda = lmbda
            #[TODO] call _kFoldSplitIndices to split indices into k groups randomly
            folds = self._kFoldSplitIndices(Y, k)
            #[TODO] for each iteration i = 1...k, train the model using lmbda
            # on k−1 folds of data. Then test with the i-th fold.
            total_error = 0
            for i in range(k):
                X_train = X.copy()
                to_excludeX = folds[i]
                X_train = np.array([elem for j, elem in enumerate(X_train) if j not in to_excludeX])
                X_val = X.copy()
                X_val = np.array(X_val)[to_excludeX]
                Y_train = Y.copy()
                Y_train = np.array([elem for j, elem in enumerate(Y_train) if j not in to_excludeX])
                self.train(X_train, Y_train)
                Y_val = Y.copy()
                Y_val = np.array(Y_val)[to_excludeX]
                total_error += 1 - self.accuracy(X_val,Y_val)
            
            #[TODO] calculate and record the cross validation error by averaging total errors
            k_fold_errors.append(total_error/k)
        return k_fold_errors

    def plotError(self, lambda_list, train_errors, val_errors, k_fold_errors):
        '''
        Produce a plot of the cost function on the training and validation sets, and the
        cost function of k-fold with respect to the regularization parameter lambda. Use this plot
        to determine a valid lambda.
        @params:
            lambda_list: a list of lambdas
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        @return:
            None
        '''
        plt.figure()
        plt.semilogx(lambda_list, train_errors, label = 'training error')
        plt.semilogx(lambda_list, val_errors, label = 'validation error')
        plt.semilogx(lambda_list, k_fold_errors, label = 'k-fold error')
        plt.xlabel('lambda')
        plt.ylabel('error')
        plt.legend()
        plt.show()