## theanoml version 2
## major revision: simplification of formulas, genearlization of optimization methods



## documents (from theano) about all these models can be found online at
## http://deeplearning.net/tutorial/intro.html

## the source code of original theano tutorial can be found at 
## https://github.com/lisa-lab/DeepLearningTutorials

## the main purpose of this library is so to wrap the theano code
## into usable format by following the sklearn interface

## Models that have been included or planned to be included are,
## Logistic Regression cg and sgd
## MLP with sgd
## Deep Convolutional Network (simplified version of LeNet5)
## Auto Encoders, Denoising Autoencoders
## Restricted Boltzmann Machines
## Deep Belief Networks

import formula
import optimize
import linear_model
import mlp
import lenet