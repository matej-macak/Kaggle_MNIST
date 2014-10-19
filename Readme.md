## MNIST Digit Recognition

### Model information
This repository contains information on how to implement a deep multilayer perceptron for [Kaggle Digit Recognizer Competition](http://www.kaggle.com/c/digit-recognizer). The model used is based on LISA Lab Theano [implementation](http://www.deeplearning.net/tutorial/) and [Ciresan et al. (2010)](http://arxiv.org/abs/1003.0358). It is a big, simple MLP model which includes random constrained (in parameters) affine transforms on data before each epoch. 

### Dataset
The model has been semi-optimized for the reduced MNIST dataset as present here. Due to relatively small numbers of digits in the training set (30,000), the affine transforms can help in artificially increasing the training set and prevent overfitting. The `preprocessing.py` contains a further possible increase by generating artificial data using font-library on the computer (Untested).

### Performance
In the default settings, test performance is ~1.1%. Only simple affine transforms have been implemented so far (implemented on notebook i7-4702HQ, GeForce 750m). Elastic distortions have been found to improve performance considerably ([Simard et al. 2003](http://research.microsoft.com/apps/pubs/?id=68920)).

### Requirements
Requirements for the script to run are basic scientific libraries `numpy`, `scipy` and `Theano` configured to run with the GPU. This was tested on a 32-bit `python 2.7.6` in Windows 8.1 environment. 

### Implementation
`main.py` contains the basic implementation parameters which can be easily modified by changing the `dict` contents. 


