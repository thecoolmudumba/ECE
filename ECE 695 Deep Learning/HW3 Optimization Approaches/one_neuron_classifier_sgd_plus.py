#!/usr/bin/env python

##  one_neuron_classifier.py

"""
A one-neuron model is characterized by a single expression that you see in the value
supplied for the constructor parameter "expressions".  In the expression supplied, the
names that being with 'x' are the input variables and the names that begin with the
other letters of the alphabet are the learnable parameters.
"""

import random
import numpy
import matplotlib.pyplot as plt

seed = 0           
random.seed(seed)
numpy.random.seed(seed)

import sys
sys.path.append("C:/Users/Sai Mudumba/Documents/PS03/ComputationalGraphPrimer-1.0.6" )
from ComputationalGraphPrimer import *

cgp = ComputationalGraphPrimer(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = 1e-3,
#               learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )


cgp.parse_expressions()

#cgp.display_network1()
cgp.display_network2()

cgp.gen_training_data()

loss_returned_SGD_plus = cgp.run_training_loop_one_neuron_model(mu=0.99)
loss_returned_SGD = cgp.run_training_loop_one_neuron_model(mu=0)

plt.plot(loss_returned_SGD, 'g', label="SGD")
plt.plot(loss_returned_SGD_plus,'b',label="SGD+")
plt.legend(loc="upper right")
plt.title("Single-Neuron Network SGD+ versus SGD Loss")
plt.show()


