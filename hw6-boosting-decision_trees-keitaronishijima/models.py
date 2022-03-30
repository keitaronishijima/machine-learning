import numpy as np
import random
import copy
import math

def node_score_error(prob):
    '''
        TODO:
        Calculate the node score using the train error of the subdataset and return it.
        For a dataset with two classes, C(p) = min{p, 1-p}
    '''
    return min(prob, 1 - prob)
    pass


def node_score_entropy(prob):
    '''
        TODO:
        Calculate the node score using the entropy of the subdataset and return it.
        For a dataset with 2 classes, C(p) = -p * log(p) - (1-p) * log(1-p)
        For the purposes of this calculation, assume 0*log0 = 0.
        HINT: remember to consider the range of values that p can take!
    '''
    l = 0
    l2 = math.log(1)
    if prob != 0 and prob!= 1:
        l = math.log(prob)
        l2 = math.log(1- prob)
    return -prob * l - (1-prob) * l2
    pass


def node_score_gini(prob):
    '''
        TODO:
        Calculate the node score using the gini index of the subdataset and return it.
        For dataset with 2 classes, C(p) = 2 * p * (1-p)
    '''
    return 2 * prob * (1-prob)
    pass



class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} # used for visualization


    def _set_info(self, gain, num_samples):
        '''
        Helper function to add to info attribute.
        You do not need to modify this. 
        '''

        self.info['gain'] = gain
        self.info['num_samples'] = num_samples


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=node_score_entropy, max_depth=40):
       # data = np.array([[1,0,1,1], [0,0,0,1], [0,1,1,0], [0,0,0,0]])
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)


    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)


    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)


    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)


    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        NOTE:
        This might be slightly different from the pruning described in lecture.
        Here we won't consider pruning a node's parent if we don't prune the node 
        itself (i.e. we will only prune nodes that have two leaves as children.)
        HINT: Think about what variables need to be set when pruning a node!
        '''
        if node is None or node.isleaf is True:
            return
        self._prune_recurs(node.left, validation_data)

        if node.left.isleaf is True and node.right.isleaf is True:
            curr_loss = self.loss(validation_data)
            # store current left and right
            r_tmp = node.right
            l_tmp = node.left
            node.right = None
            node.left = None
            node.isleaf = True
            new_loss = self.loss(validation_data)
            if new_loss > curr_loss:
                node.right = r_tmp
                node.left = l_tmp
                node.isleaf = False
        self._prune_recurs(node.right, validation_data)
        pass


    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion if:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node reaches the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf and 
              False if the node is not a leaf.
            - A label, indicating the label of the leaf (or the label the node would 
              be if we were to terminate at that node). If there is no data left, you
              can return either label at random.
        '''

        majority = 0
        zero_count = 0
        one_count = 0 
        for i in range(len(data)):
            if data[i][0] == 0:
                zero_count += 1
            else:
                one_count += 1
        if zero_count < one_count:
            majority = 1


        # dataset is empty
        if len(data) == 0:
            return True, round(random.random())
        
        # No more indices to split
        if len(indices) == 0:
            return True, majority
    
        # All instance same class
        is_one_in = False
        is_zero_in = False        
        for i in range(len(data)):
            if data[i][0] == 1:
                is_one_in = True
            if data[i][0] == 0:
                is_zero_in = True
        
        if is_one_in == False or is_zero_in == False:
            return True, majority
        
        # max depth
        if node.depth == self.max_depth:
            return True, majority
    
        # if not leaf
        return False, majority


    def _split_recurs(self, node, data, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.

        First use _is_terminal() to check if the node needs to be split.
        If so, select the column that has the maximum infomation gain to split on.
        Store the label predicted for this node, the split column, and use _set_info()
        to keep track of the gain and the number of datapoints at the split.
        Then, split the data based on its value in the selected column.
        The data should be recursively passed to the children.
        '''
        #print(node.depth)
        node.isleaf, node.label = self._is_terminal(node,data,indices)
        if node.isleaf:
            return

        # Find the information maximizing value and column
        max_gain = -1000000000
        max_column = -1
        for i in range(len(indices)):
            if max_gain <= self._calc_gain(data, indices[i], self.gain_function):
                max_gain = self._calc_gain(data, indices[i], self.gain_function)
                max_column = indices[i]
  
        # Predicted label for the node
        # dummy, label = self._is_terminal(node,data,indices)
        node.index_split_on = max_column
        node._set_info(max_gain, len(data))

        #Split the data into right and left
        right_data = []
        left_data = []
        for i in range(len(data)):
            #print("i and max_column are", i, max_column)
            #print("___________")
            if data[i][max_column] == 1:
                right_data.append(data[i])
            else:
                left_data.append(data[i])

        # Create left and right nodes
        right_node = Node(depth = node.depth + 1)
        left_node = Node(depth = node.depth + 1)
        node.right = right_node
        node.left = left_node
        indices_copy = indices.copy()
        indices_copy.remove(max_column)
        self._split_recurs(node.right, right_data, indices_copy)
        self._split_recurs(node.left, left_data, indices_copy)

    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - P[x_i=True] * C(P[y=1|x_i=True]) - P[x_i=False] * C(P[y=0|x_i=False])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        if (len(data) == 0):
            return 
        label_count = 0
        for i in range(len(data)) :
            if data[i][0] == 1:
                label_count += 1
        right = []
        left = []
        for i in range(len(data)):
            if data[i][split_index] == 1:
                right.append(data[i])
            else:
                left.append(data[i])
        prob_right = len(right) / len(data)
        prob_left = len(left) / len(data)
        label_count_right = 0
        label_count_left = 0
        for i in range(len(right)):
            if right[i][0] == 1:
                label_count_right += 1
        for i in range(len(left)):
            if left[i][0] == 0:
                label_count_left += 1
        probs_root = label_count / len(data)
        if len(right) != 0:
            probs_right = label_count_right / len(right)
        else:
            probs_right = 0.5
        if len(left) != 0:
            probs_left = label_count_left / len(left)
        else:
            probs_left = 0.5
        gain = gain_function(probs_root) - prob_right * gain_function(probs_right) - prob_left * gain_function(probs_left)
        return gain
    

    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = {:d}; gain = {:f}; number of samples = {:d}'.format(node.index_split_on, node.info['gain'], node.info['num_samples'])
            left = indent + '0 -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + '1 -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')


    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)


    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if not row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
