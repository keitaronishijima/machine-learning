import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from models import DecisionTree, node_score_error, node_score_entropy, node_score_gini


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)
    plt.show()

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)
    dt_non_prune = DecisionTree(train_data, None, node_score_error, 15)
    dt_prune = DecisionTree(train_data, validation_data, node_score_error, 15)
    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    #      (b) Print average test loss (not-pruned)
    #      (c) Print average training loss (pruned)
    #      (d) Print average test loss (pruned)
    sum = 0
    for i in train_data:
        sum = sum + dt_non_prune.loss(train_data)
    print(sum/len(train_data))
    sum = 0
    for i in test_data:
        sum = sum + dt_non_prune.loss(test_data)
    print(sum/len(train_data))
    sum = 0
    for i in train_data:
        sum = sum + dt_prune.loss(train_data)
    print(sum/len(train_data))
    sum = 0
    for i in test_data:
        sum = sum + dt_prune.loss(test_data)
    print(sum/len(train_data))
    
    
    dt_non_prune = DecisionTree(train_data, None, node_score_entropy, 15)
    dt_prune = DecisionTree(train_data, validation_data, node_score_entropy, 15)

    sum = 0
    for i in test_data:
        sum = sum + dt_non_prune.loss(test_data)
    print(sum/len(train_data))
    sum = 0
    for i in train_data:
        sum = sum + dt_prune.loss(train_data)
    print(sum/len(train_data))
    sum = 0
    for i in test_data:
        sum = sum + dt_prune.loss(test_data)
    print(sum/len(train_data))
    dt_non_prune = DecisionTree(train_data, None, node_score_gini, 15)
    dt_prune = DecisionTree(train_data, validation_data, node_score_gini, 15)
    sum = 0
    for i in train_data:
        sum = sum + dt_non_prune.loss(train_data)
    print(sum/len(train_data))
    sum = 0
    for i in test_data:
        sum = sum + dt_non_prune.loss(test_data)
    print(sum/len(train_data))
    sum = 0
    for i in train_data:
        sum = sum + dt_prune.loss(train_data)
    print(sum/len(train_data))
    sum = 0
    for i in test_data:
        sum = sum + dt_prune.loss(test_data)
    print(sum/len(train_data))



    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!

def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    #explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')
    plt.show()

main()
