#!/usr/bin/env python

##  multi_neuron_classifier.py

"""
The main point of this script is to demonstrate saving the information during the
forward propagation of data through a neural network and using that information for
backpropagating the loss and for updating the values for the learnable parameters.  The
script uses the following 4-2-1 network layout, with 4 nodes in the input layer, 2 in
the hidden layer and 1 in the output layer as shown below:


                               input

                                 x                                             x = node

                                 x         x|                                  | = sigmoid activation
                                                     x|
                                 x         x|

                                 x

                             layer_0    layer_1    layer_2


To explain what information is stored during the forward pass and how that
information is used during the backprop step, see the comment blocks associated with
the functions

         forward_prop_multi_neuron_model()   
and
         backprop_and_update_params_multi_neuron_model()

Both of these functions are called by the training function:

         run_training_loop_multi_neuron_model()

"""

import random
import numpy
import matplotlib.pyplot as plt
seed = 0           
random.seed(seed)
numpy.random.seed(seed)

from ComputationalGraphPrimer import *

cgp = ComputationalGraphPrimer(
               num_layers = 3,
               layers_config = [4,2,1],                         # num of nodes in each layer
               expressions = ['xw=ap*xp+aq*xq+ar*xr+as*xs',
                              'xz=bp*xp+bq*xq+br*xr+bs*xs',
                              'xo=cp*xw+cq*xz'],
               output_vars = ['xo'],
               dataset_size = 5000,
               learning_rate = 1e-3,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )

cgp.parse_multi_layer_expressions()

#cgp.display_network1()
cgp.display_network2()

cgp.gen_training_data()

loss_SGD_plus = cgp.run_training_loop_multi_neuron_model(mu=0.99)
loss_SGD = cgp.run_training_loop_multi_neuron_model(mu=0)

plt.plot(loss_SGD, 'g', label="SGD")
plt.plot(loss_SGD_plus,'b',label="SGD+")
plt.legend(loc="upper right")
plt.title("Multi-Neuron Network SGD+ versus SGD Loss")
plt.show()
