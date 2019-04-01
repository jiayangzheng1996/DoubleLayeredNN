import numpy as np
from io import StringIO

NUM_FEATURES = 0 #this variable determines the number of input units in the network
DATA_PATH = "" #put the data file here

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 instead, because sigmoid's range is 0-1. this does not matter if your label is 0 and 1.
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32).reshape(len(xs),NUM_FEATURES,1) #returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):
    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
    else:
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES) #bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1) #add bias column

    #w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). The last column is the bias weights.
    model = (w1,w2)
    return model

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0-x)

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, test_ys, test_xs, tlist, dlist, args):
    for iteration in range(args.iterations):
        w1, w2 = extract_weights(model)
        for i in range(len(train_ys)): #each data point i
            y = train_ys[i]
            x = train_xs[i] #124*1 matrix
            #feed forward
            a1 = np.dot(w1,x)
            z1 = sigmoid(a1)
            #create z1 with bias
            z1b = np.ones((len(z1)+1,len(z1[0])))
            z1b[:-1,:] = z1

            output = np.dot(w2,z1b)
            y_ = sigmoid(output)

            #backprop
            delta2 = (y_ - y) * sigmoid_derivative(y_)
            dE2 = delta2*z1.T
            dE2b = np.zeros((len(dE2),len(dE2[0])+1))
            dE2b[:,:-1] = dE2
            dE2b[:,-1] = delta2.T
            w2nb = w2[:,:-1]
            delta1 = np.multiply((w2nb.T*delta2),sigmoid_derivative(z1))
            a0 = x[:-1,:]
            dE1 = delta1*a0.T
            dE1b = np.zeros((len(dE1), len(dE1[0]) + 1))
            dE1b[:,:-1] = dE1
            dE1b[:,-1] = delta1.T

            #update w1
            w1 -= args.lr*dE1b

            #update w2
            w2 -= args.lr*dE2b

        model = (w1, w2)

        #plot variable
        taccuracy = test_accuracy(model, test_ys, test_xs)
        tlist[iteration] = taccuracy
        daccuracy = test_accuracy(model, dev_ys, dev_xs)
        dlist[iteration] = daccuracy
        
        #convergence check
        if not args.nodev:
            accuracy_new = daccuracy
            if np.abs(accuracy_new - accuracy_old) < 0.001:
                break
            accuracy_old = accuracy_new

    return model

def test_accuracy(model, test_ys, test_xs):
    accuracy = 0.0
    for i in range(len(test_ys)):  # each data point i
        y = test_ys[i]
        x = test_xs[i]  # 124*1 matrix
        # feed forward
        w1, w2 = extract_weights(model)  # w1 is 1*124; w2 is 1*3
        a1 = np.dot(w1,x)
        z1 = sigmoid(a1)
        z1b = np.ones((len(z1) + 1, len(z1[0])))
        z1b[:-1, :] = z1
        y_ = sigmoid(np.dot(w2, z1b))

        if y_ >= 0.5:
            output = 1
        else:
            output = 0

        if output == y:
            accuracy += 1.0
    accuracy = accuracy / len(test_ys)

    return accuracy

def extract_weights(model):
    w1 = None
    w2 = None
    w1 = model[0]
    w2 = model[1]
    return w1, w2

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, help='Training data file.')
    parser.add_argument('--dev_file', type=str, help='Dev data file.')
    parser.add_argument('--test_file', type=str, help='Test data file.')


    args = parser.parse_args()

    """
    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """

    tlist = [0.0] * (args.iterations)
    dlist = [0.0] * (args.iterations)

    train_ys, train_xs = parse_data(args.train_file)
    dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, test_ys, test_xs, tlist, dlist, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))
    
    #plotting graph
    import matplotlib.pyplot as plt
    import matplotlib.patches as patch
    plt.plot(tlist, color='blue')
    plt.plot(dlist, color='green')
    test_patch = patch.Patch(color='blue', label='test data')
    development_patch = patch.Patch(color='green', label='development data')
    plt.title('NeuralNetwork')
    plt.ylabel('Accuracy')
    plt.xlabel('Num of iteration')
    plt.xlim(left=1)
    plt.xlim(right=(args.iterations + 1))
    plt.legend(handles=[test_patch, development_patch])
    plt.show()


if __name__ == '__main__':
    main()
