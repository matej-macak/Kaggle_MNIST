import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
import pandas as pd

class LogisticRegression(object):
    """Multi-class Logistic Regression Class
    """

    def __init__(self, input, n_in, n_out,load=None):
        """ Initialize the parameters of the logistic regression

        input (theano.tensor.TensorType) - symbolic value describing the input
        of the architecture (one minibatch)

        n_in (int) - number of input units (from preceding layer)

        n_out (int) - number of output units 
        
        load (str) - path to parameters in case interrupt occured

        """
        if not load:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            self.W = theano.shared(value=np.zeros((n_in, n_out),
                                                     dtype=theano.config.floatX),
                                    name='W', borrow=True)
            # initialize the baises b as a vector of n_out 0s
            self.b = theano.shared(value=np.zeros((n_out,),
                                                     dtype=theano.config.floatX),
                                   name='b', borrow=True)
        else:
            # load file values
            f = file(load,'rb')
            obj = []
            for i in range(2):
                obj.append(cPickle.load(f))
            
            f.close()         
            W, b = obj[0].get_value(), obj[1].get_value()
            
            self.W = theano.shared(value=W, name='W', borrow=True)
            self.b = theano.shared(value=b, name='b', borrow=True)


        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        """

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Returns a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        y (theano.tensor.TensorType) - vector that gives correct label for each
        example
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



def load_data(dataset, split = [5,1,1], permute = True):
    ''' Loads the dataset

    dataset (str) - path to dataset
    '''

    def shared_dataset(data, borrow=True):
        """ Function that loads the dataset into shared variables

        """
        data_x, data_y = data
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        # The reason for cast is to convert into 'int32' from float which
        # is required by GPU
        return shared_x, T.cast(shared_y, 'int32')

    def get_indices(shape, split = split, permute = True):
        """
        Splits indices based on the list provided into:
        
        train data - split[0]
        valid data - split[1]
        test data - split[2]
        """        
        
        if permute == True:
            #Permute the data vector
            perm = np.random.permutation(np.arange(shape))
        else:
            #Or just use the standard ones
            perm = np.arange(shape)
        
        index = 0
        indices = [] 
        # Split the indices based on the list provided
        for num in split:
            index_end = index+(shape/np.sum(split))*num
            indices.append(perm[index:index_end])
            index += (shape/np.sum(split))*num
            
        return indices

    old = pd.read_csv(dataset).values
    
    imgs = np.zeros(old[:,1:].shape,dtype=np.float32)
    labels = np.zeros(old[:,0].shape,dtype=np.int64)
    for img in range(old.shape[0]):
        imgs[img,:] = np.float32(old[img,1:]/255.0)
        labels[img] = old[img,0]
                
    indices = get_indices(imgs.shape[0], split=split)
        
    train_set_x, train_set_y = shared_dataset((imgs[indices[0],:],labels[indices[0]]))      
    test_set_x, test_set_y = shared_dataset((imgs[indices[1],:],labels[indices[1]]))
    valid_set_x, valid_set_y = shared_dataset((imgs[indices[2],:],labels[indices[2]]))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval