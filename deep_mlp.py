__docformat__ = 'restructedtext en'

import cPickle
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

from initialize_mlp import LogisticRegression, load_data
from preprocessing import batch_transforms

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        rng (np.random.RandomState) - a random number generator for weight
        initialization
        
        input (theano.tensor.dmatrix) - a symbolic tensor of shape (n_examples, n_in)
        
        n_in (int) - dimensionality of input
        
        n_out (int) - number of hidden units

        activation(theano.Op or function) - non-linearity applied to hidden layer

        """
        self.input = input

        # Initialization of weights based on the deeplearning.net guide for MLPs
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        # Initialize bias term
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
 
       # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """
    The base MLP class
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        rng - random state generator
        
        n_in - shape of the input        
        
        n_hidden - number of hidden units passed as a list with first value
        corresponding to first layer
        
        n_out - number of output units from the logRegressionLayer
        
        """

        self.layers = []        
        for hidden in range(len(n_hidden)):
            #If first layer, set n_in to n_in
            if hidden == 0:
                self.layers.append(HiddenLayer(rng=rng, input = input,
                                               n_in=n_in, n_out=n_hidden[hidden],
                                               activation=T.tanh))
            elif hidden > 0:
                self.layers.append(HiddenLayer(rng=rng, input = self.layers[hidden-1].output,
                                               n_in = n_hidden[hidden-1],n_out = n_hidden[hidden],
                                               activation = T.tanh))
                                   

        # Output layer
        self.logRegressionLayer = LogisticRegression(input = self.layers[-1].output,
                                                     n_in = n_hidden[-1],
                                                     n_out = n_out)

        # L1 norm
        self.L1 = np.sum([abs(layer.W).sum() for layer in self.layers]) \
                + abs(self.logRegressionLayer.W).sum()
              
        # L2 square
        self.L2_sqr = np.sum([(layer.W**2).sum() for layer in self.layers]) \
                + (self.logRegressionLayer.W**2).sum()

        # Negative log likelihood as based on the logistic output
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # Errors of the model
        self.errors = self.logRegressionLayer.errors

        # Parameters of the model
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
        self.params += self.logRegressionLayer.params



def train_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', permute_data = False, batch_size=20, n_hidden=[2000,1500,1000],
             output_folder = os.getcwd()+'/Parameters/', base_name = 'deep_mlp',
             load_model_name = None, **kwargs):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    learning rate (float) - factor for stochastic gradient
    
    L1_reg (float) - L1 norm regularization weight

    L2_reg (float) - L2 norm regularization weight
    
    n_epochs (int) - max number of epochs to run the optimizer
    
    n_hidden (list) - constructs MLP based on the list of ints with first entry
    being first layer, etc.
    
    permute_data (boolean) - if True, permutes the order before splitting into train,
    test and validate sets.
    
    batch_size (int) - number of batches to split the dataset into.
    
    output_folder (str) - folder for saving the parameters of model. The default
    value is Parameters folder where the script is located.
    
    base_name (str) - base name of parameters to be saved
    
    load_model_name (None, str) - file path for loading previous model variables
    if interrupted
    
    transforms (dictionary, optional) - accepts dictionary of affine transform
    values for shearing, scaling and rotatin
    
    early_stopping (dictionary, optional) - enables overriding of default
    early-stopping parameters

   """

    # Permute data if necessary
    datasets = load_data(dataset, permute = permute_data)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of ints

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=28 * 28,
                     n_hidden=n_hidden, n_out=10)

    # The cost to minimize expressed symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr

    # Theano function to calculate test error rate
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    # Theano function to calculate validation error rate
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # Theano model to calculate cost of the training and update based on the
    # rules in updates list
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'


    if not kwargs['early_stopping']:
        # If user does not specify early stopping parameters
        
        # early-stopping parameters
        patience = 10000  # look at this many examples regardless
        # Wait for this much longer when a new best is found    
        patience_increase = 2  
        #A relative improvement considered significant
        improvement_threshold = 0.995  
    else:
        patience = kwargs['early_stopping']['patience']
        patience_increase = kwargs['early_stopping']['patience_increase']
        improvement_threshold = kwargs['early_stopping']['improvement_threshold']
        
    
    # Go through this many cases before checking the network on validation set
    validation_frequency = min(n_train_batches, patience / 2)

    # Load parameters of models if interrupted otherwise state default values
    if not load_model_name:
        valid_params = None
        test_params = None
        best_validation_loss = np.inf
        best_test_loss = np.inf
        valid_iter = 0
        test_iter = 0
        test_score = 0.
        epoch = 0
    else:
        epoch = 0
        model = load_model(load_model_name)
        epoch = model['epoch']
        valid_params = model['valid_params']
        test_params = model['test_params']
        best_validation_loss = model['best_validation_loss']
        best_test_loss = model['best_test_loss']
        valid_iter = model['valid_iter']
        test_iter = model['test_iter']
        test_score = model['test_score']
        epoch = model['epoch']
        if 'transforms' in model['other'].keys(): transforms = model['other']['transforms']
        if 'early_stopping' in model['other'].keys(): early_stopping = model['other']['early_stopping']
    
        for param in range(len(model['valid_params'])):
            classifier.params[param].set_value(
            model['valid_params'][param].get_value(borrow=True),borrow=True)
            
    
    start_time = time.clock()    
    done_looping = False

    #Prepare transform parameters and data
    base_data = np.float32(train_set_x.get_value(borrow=True))
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        if kwargs['transforms']:
            train_set_x.set_value(batch_transforms(base_data, **kwargs['transforms']),borrow=True)

        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    #Update best validation test scores
                    best_validation_loss = this_validation_loss
                    valid_iter = iter
                    valid_params = [obj for obj in classifier.params]                                        

                    # test it on the test set and update test params if better
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    
                    if test_score < best_test_loss:
                        best_test_loss = test_score
                        test_iter = iter
                        test_params = [obj for obj in classifier.params]

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    break
        
        # Save at each 100th epoch
        if epoch%1 == 0:
            save_model(epoch=epoch, output_folder=output_folder, base_name = base_name,
               n_hidden = n_hidden, valid_params = valid_params, test_params = test_params,
               best_validation_loss = best_validation_loss, best_test_loss = best_test_loss,
               valid_iter = valid_iter, test_iter = test_iter, test_score = test_score)

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., valid_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    save_model(epoch=epoch, output_folder=output_folder, base_name = base_name,
               n_hidden = n_hidden, valid_params = valid_params, test_params = test_params,
               best_validation_loss = best_validation_loss, best_test_loss = best_test_loss,
               valid_iter = valid_iter, test_iter = test_iter, test_score = test_score)


def save_model(epoch = 0, output_folder='/Temp/Machine_Learning/', base_name = 'deep_mlp', 
               n_hidden = [2000,1500,1000], valid_params = None, test_params = None,
               best_validation_loss = np.inf, best_test_loss = np.inf,
               valid_iter = 0, test_iter = 0, test_score = 0, **kwargs):
                   
    model = {}
    model['epoch'] = epoch
    model['n_hidden'] = n_hidden
    model['valid_params'] = [obj for obj in valid_params]
    model['test_params'] = [obj for obj in test_params]
    model['best_validation_loss'] = best_validation_loss
    model['best_test_loss'] = best_test_loss
    model['valid_iter'] = valid_iter
    model['test_iter'] = test_iter
    model['test_score'] = test_score
    model['other'] = kwargs
    
    f = open(output_folder+base_name+str(int(epoch/100))+".pkz",'wb')
    cPickle.dump(model,f,protocol=cPickle.HIGHEST_PROTOCOL)
                
    f.close()

def load_model(file_name):
    """ 
    Re-loads model parameters after interruption.
    """        
    
    f = open(file_name, 'rb')
    model = cPickle.load(f)  
    f.close()
    
    return model


if __name__ == '__main__':
    transforms = {'scale':True, 'shear':False, 'rotate':True,
                  'scale_interval':(0.9,1.1), 'rotate_interval':(-7.5,7.5),
                  'shear_interval':(-0.05,0.05)}
    
    early_stopping = {'patience':10000, 'patience_increase':2,
                      'improvement_threshold':0.995}    
    
    train_mlp(dataset = "/Users/Matej/Dropbox/Programming/Python/Machine_Learning/MNIST/Data/train.csv",
             transforms = transforms, early_stopping = early_stopping,
             load_model_name = "/Temp/Machine_Learning/deep_net_4_4.pkz")