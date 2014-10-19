import cPickle
import pandas as pd

import numpy as np

import theano
import theano.tensor as T
from initialize_mlp import LogisticRegression, load_data
from deep_mlp import MLP

def load_data(path = "/Users/Matej/Dropbox/Programming/Python/Machine_Learning/MNIST/Data/test.csv"):
    
    data = np.float64(pd.read_csv(path).values)/255.0
    
    return data

def load_model(file_name):
    """ 
    Re-loads model parameters after interruption.
    """        
    
    f = open(file_name, 'rb')
    model = cPickle.load(f)  
    f.close()
    
    return model

def generate_predictions(data, model_name, output = '/Temp/Machine_Learning/output.csv',step = 1000):
    model = load_model(model_name)

    # Initiate variables
    index = T.lscalar()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    # Put the data into shared variable
    classify_set = theano.shared(np.asarray(data,
                                               dtype=theano.config.floatX),
                                 borrow=True)

    rng = np.random.RandomState(1234)

    # Initiate classifier based on model parameters
    classifier = MLP(rng=rng, input=x, n_in=28 * 28,
                     n_hidden=model['n_hidden'], n_out=10)
                     
    # Load up parameters of the model into the classifier class
    for param in range(len(model['valid_params'])):
        classifier.params[param].set_value(
        model['valid_params'][param].get_value(borrow=True),borrow=True)

    # Setup classification theano function
    classify = theano.function(inputs=[index],
        outputs=classifier.logRegressionLayer.y_pred,
        givens={
            x: classify_set[index*step:(index+1)*step]})

    y_pred = np.zeros(data.shape[0])    
    for idx in range(int(data.shape[0]/step)):
        y_pred[idx*step:(idx+1)*step] = classify(idx)    
        
    y = np.int64(y_pred)
    to_df = np.int64(np.vstack((np.arange(y.shape[0])+1,y)))
    df = pd.DataFrame(to_df.T,columns=['ImageId','Label'])
    df.to_csv(output,index=False)
    

data = load_data()
generate_predictions(data, "/Temp/Machine_Learning/deep_net_4_4.pkz")





