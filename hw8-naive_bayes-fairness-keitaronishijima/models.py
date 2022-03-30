import numpy as np

class NaiveBayes(object):
    """ Bernoulli Naive Bayes model
    
    ***DO NOT CHANGE the following attribute names (to maintain autograder compatiblity)***
    
    @attrs:
        n_classes:    the number of classes
        attr_dist:    a 2D (n_classes x n_attributes) NumPy array of the attribute distributions
        label_priors: a 1D NumPy array of the priors distribution
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes model with n_classes. """
        self.n_classes = n_classes
        self.attr_dist = None
        self.label_priors = None

    def train(self, X_train, y_train):
        """ Trains the model, using maximum likelihood estimation.
        @params:
            X_train: a 2D (n_examples x n_attributes) numpy array
            y_train: a 1D (n_examples) numpy array
        @return:
            a tuple consisting of:
                1) a 2D numpy array of the attribute distributions
                2) a 1D numpy array of the priors distribution
        """

        # TODO
        n = len(X_train)
        zeros = 0
        ones = 0
        for i in range(n):
            if y_train[i] == 0:
                zeros += 1
            else:
                ones += 1
        att = len(X_train[0])
        self.attr_dist = np.full((att, self.n_classes), 1.0)
        self.label_priors = np.ones((self.n_classes))
        # Step 1: Count the fraction of times each class appears in training data
        for i in range(n):
            self.label_priors[y_train[i]] += 1
        bot = self.n_classes + n
        for i in range(self.n_classes):
            self.label_priors[i] = self.label_priors[i] / bot
            
        # Step 2: Estimate the attribute distributions
        for i in range(self.n_classes):
            for j in range(att):
                for k in range(n):
                    if X_train[k][j] == 1 and y_train[k] == i:
                        self.attr_dist[j][i] += 1.0
        for i in range(att):
            for j in range(self.n_classes):
                if j == 0:
                    self.attr_dist[i][j] = self.attr_dist[i][j]/(zeros+2)
                else:
                    self.attr_dist[i][j] = self.attr_dist[i][j]/(ones+2)
        self.attr_dist = self.attr_dist.T
        return (self.attr_dist, self.label_priors)
                
    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.
            Remember to convert to log space to avoid overflow/underflow
            errors!

        @params:
            inputs: a 2D NumPy array containing inputs
        @return:
            a 1D numpy array of predictions
        """

        # TODO
        # for i in range(len(inputs)):
        #     if inputs[i][0] == 0:
        #         for j in range(1,len(inputs[i])):
        #             inputs[i][j] = 1 - inputs[i][j]
        self.attr_dist = self.attr_dist.T
        ret = np.zeros((len(inputs)))
        # for i in range(len(inputs[0]) - 1):
        #     ret[i] = np.prod(inputs[:, 1:],axis = 0)
        for i in range(len(inputs)):
            example = np.array([inputs[i]]).reshape((len(inputs[i]),1))
            mat = np.hstack((example, self.attr_dist))
            for j in range(len(mat)):
                if mat[j][0] == 0:
                    for k in range(1,len(mat[j])):
                        mat[j][k] = 1 - mat[j][k]
            tmp = np.array([0,0])
            tmp = np.prod(mat[:, 1:],axis = 0) # 62, 2
            for j in range(2):
                if j == 0:
                    tmp[0] = tmp[0] * self.label_priors[0]
                else:
                    tmp[j] = tmp[j] * self.label_priors[1]
            ret[i] = np.argmax(tmp)
        self.attr_dist = self.attr_dist.T
        return ret
    def accuracy(self, X_test, y_test):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            X_test: a 2D numpy array of examples
            y_test: a 1D numpy array of labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """

        # TODO
        pred = self.predict(X_test)
        n = len(X_test)
        count = 0
        for i in range(n):
            if y_test[i] == pred[i]:
                count+=1
        return count/n
    def print_fairness(self, X_test, y_test, x_sens):
        """ 
        ***DO NOT CHANGE what we have implemented here.***
        
        Prints measures of the trained model's fairness on a given dataset (data).

        For all of these measures, x_sens == 1 corresponds to the "privileged"
        class, and x_sens == 0 corresponds to the "disadvantaged" class. Remember that
        y == 1 corresponds to "good" credit. 

        @params:
            X_test: a 2D numpy array of examples
            y_test: a 1D numpy array of labels
            x_sens: a numpy array of sensitive attribute values
        @return:

        """
        predictions = self.predict(X_test)

        # Disparate Impact (80% rule): A measure based on base rates: one of
        # two tests used in legal literature. All unprivileged classes are
        # grouped together as values of 0 and all privileged classes are given
        # the class 1. . Given data set D = (S,X,Y), with protected
        # attribute S (e.g., race, sex, religion, etc.), remaining attributes X,
        # and binary class to be predicted Y (e.g., “will hire”), we will say
        # that D has disparate impact if:
        # P[Y^ = 1 | S != 1] / P[Y^ = 1 | S = 1] <= (t = 0.8). 
        # Note that this 80% rule is based on US legal precedent; mathematically,
        # perfect "equality" would mean

        di = np.mean(predictions[np.where(x_sens==0)])/np.mean(predictions[np.where(x_sens==1)])
        print("Disparate impact: " + str(di))

        # Group-conditioned error rates! False positives/negatives conditioned on group
        
        pred_priv = predictions[np.where(x_sens==1)]
        pred_unpr = predictions[np.where(x_sens==0)]
        y_priv = y_test[np.where(x_sens==1)]
        y_unpr = y_test[np.where(x_sens==0)]

        # s-TPR (true positive rate) = P[Y^=1|Y=1,S=s]
        priv_tpr = np.sum(np.logical_and(pred_priv == 1, y_priv == 1))/np.sum(y_priv)
        unpr_tpr = np.sum(np.logical_and(pred_unpr == 1, y_unpr == 1))/np.sum(y_unpr)

        # s-TNR (true negative rate) = P[Y^=0|Y=0,S=s]
        priv_tnr = np.sum(np.logical_and(pred_priv == 0, y_priv == 0))/(len(y_priv) - np.sum(y_priv))
        unpr_tnr = np.sum(np.logical_and(pred_unpr == 0, y_unpr == 0))/(len(y_unpr) - np.sum(y_unpr))

        # s-FPR (false positive rate) = P[Y^=1|Y=0,S=s]
        priv_fpr = 1 - priv_tnr 
        unpr_fpr = 1 - unpr_tnr 

        # s-FNR (false negative rate) = P[Y^=0|Y=1,S=s]
        priv_fnr = 1 - priv_tpr 
        unpr_fnr = 1 - unpr_tpr

        print("FPR (priv, unpriv): " + str(priv_fpr) + ", " + str(unpr_fpr))
        print("FNR (priv, unpriv): " + str(priv_fnr) + ", " + str(unpr_fnr))
    
    
        # #### ADDITIONAL MEASURES IF YOU'RE CURIOUS #####

        # Calders and Verwer (CV) : Similar comparison as disparate impact, but
        # considers difference instead of ratio. Historically, this measure is
        # used in the UK to evalutate for gender discrimination. Uses a similar
        # binary grouping strategy. Requiring CV = 1 is also called demographic
        # parity.

        cv = 1 - (np.mean(predictions[np.where(x_sens==1)]) - np.mean(predictions[np.where(x_sens==0)]))

        # Group Conditioned Accuracy: s-Accuracy = P[Y^=y|Y=y,S=s]

        priv_accuracy = np.mean(predictions[np.where(x_sens==1)] == y_test[np.where(x_sens==1)])
        unpriv_accuracy = np.mean(predictions[np.where(x_sens==0)] == y_test[np.where(x_sens==0)])

        return predictions
