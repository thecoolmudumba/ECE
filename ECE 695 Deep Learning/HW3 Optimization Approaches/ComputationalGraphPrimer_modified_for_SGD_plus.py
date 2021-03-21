__version__   = '1.0.6'
__author__    = "Avinash Kak (kak@purdue.edu)"
__date__      = '2021-February-22'   
__url__       = 'https://engineering.purdue.edu/kak/distCGP/ComputationalGraphPrimer-1.0.6.html'
__copyright__ = "(C) 2021 Avinash Kak. Python Software Foundation."

__doc__ = '''

ComputationalGraphPrimer.py

Version: ''' + __version__ + '''
   
Author: Avinash Kak (kak@purdue.edu)

Date: ''' + __date__ + '''


@title
CHANGE LOG:

  Version 1.0.6:

    This version includes a demonstration of how to extend PyTorch's
    Autograd class if you wish to customize how the learnable parameters
    are updated during backprop on the basis of the data conditions
    discovered during the forward propagation.  Previously this material
    was in the DLStudio module.

  Version 1.0.5:

    I have been experimenting with different ideas for increasing the
    tutorial appeal of this module.  (That is the reason for the jump in
    the version number from 1.0.2 to the current 1.0.5.)  The previous
    public version provided a simple demonstration of how one could forward
    propagate data through a DAG (Directed Acyclic Graph) while at the same
    compute the partial derivatives that would be needed subsequently
    during the backpropagation step for updating the values of the
    learnable parameters.  In 1.0.2, my goal was just to illustrate what
    was meant by a DAG and how to use such a representation for forward
    data flow and backward parameter update.  Since I had not incorporated
    any nonlinearities in such networks, there was obviously no real
    learning taking place.  That fact was made evident by a plot of
    training loss versus iterations.

    To remedy this shortcoming of the previous public-release version, the
    current version introduces two special cases of networks --- a
    one-neuron network and a multi-neuron network --- for experimenting
    with forward propagation of data while calculating the partial
    derivatives needed later, followed by backpropagation of the prediction
    errors for updating the values of the learnable parameters. In both
    cases, I have used the Sigmoid activation function at the nodes. The
    partial derivatives that are calculated in the forward direction are
    based on analytical formulas at both the pre-activation point for data
    aggregation and the post-activation point.  The forward and backward
    calculations incorporate smoothing of the prediction errors and the
    derivatives over a batch as required by stochastic gradient descent.

  Version 1.0.2:

    This version reflects the change in the name of the module that was
    initially released under the name CompGraphPrimer with version 1.0.1


@title
INTRODUCTION:

    This module was created with a modest goal in mind: its purpose being
    merely to serve as a prelude to discussing automatic calculation of the
    gradients of the loss with respect to the learnable parameters in
    modern Python based platforms for deep learning.

    Most students taking classes on deep learning focus on just using the
    tools provided by platforms such as PyTorch without any understanding
    of how the tools really work.  Consider, for example, Autograd --- a
    module that is at the heart of PyTorch --- for automatic
    differentiation of the loss at the output of a neural layer with
    respect to all the learnable parameters that go into calculating that
    output. With no effort on the part of the programmer, and through the
    functionality built into the "torch.Tensor" class, the Autograd module
    keeps track of a tensor through all calculations involving the tensor
    and computes the partial derivatives of the output of the layer with
    respect to the parameters stored in the tensor.  These derivatives are
    subsequently used to update the values of the learnable parameters
    during the backpropagation step.

    Now imagine a beginning student trying to make sense of the following
    excerpts from the official PyTorch documentation related to Autograd:

       "Every operation performed on Tensors creates a new function object,
        that performs the computation, and records that it happened. The
        history is retained in the form of a DAG of functions, with edges
        denoting data dependencies (input <- output). Then, when backward
        is called, the graph is processed in the topological ordering, by
        calling backward() methods of each Function object, and passing
        returned gradients on to next Functions."

        and

       "Check gradients computed via small finite differences against
        analytical gradients w.r.t. tensors in inputs that are of floating
        point type and with requires_grad=True."

    There is a lot going on here: Why do we need to record the history of
    the operations carried out on a tensor?  What is a DAG?  What are the
    returned gradients?  Gradients of what?  What are the small finite
    differences?  Analytical gradients of what? etc. etc.

    This module has three goals:

    1)    To introduce you to forward and backward dataflows in a Directed
          Acyclic Graph (DAG).

    2)    To extend the material developed for the first goal with simple
          examples of neural networks for demonstrating the forward and
          backward dataflows for the purpose of updating the learnable
          parameters.  This part of the module also includes a comparison
          of the performance of such networks with those constructed using
          torch.nn components.

    3)    To explain how the behavior of PyTorch's Autograd class can 
          be customized to your specific data needs by extending that 
          class.


    GOAL 1:           

    The first goal of this Primer is to introduce you to forward and
    backward dataflows in a general DAG. The acronym DAG stands for
    Directed Acyclic Graph. Just for the educational value of playing with
    dataflows in DAGs, this module allows you to create a DAG of variables
    with a statement like

               expressions = ['xx=xa^2',
                              'xy=ab*xx+ac*xa',
                              'xz=bc*xx+xy',
                              'xw=cd*xx+xz^3']

    where we assume that a symbolic name that beings with the letter 'x' is
    a variable, all other symbolic names being learnable parameters, and
    where we use '^' for exponentiation. The four expressions shown above
    contain five variables --- 'xx', 'xa', 'xy', 'xz', and 'xw' --- and
    four learnable parameters: 'ab', 'ac', 'bc', and 'cd'.  The DAG that is
    generated by these expressions looks like:

                    
             ________________________________ 
            /                                 \
           /                                   \
          /             xx=xa**2                v                                               
       xa --------------> xx -----------------> xy   xy = ab*xx + ac*xa
                          | \                   |                                   
                          |  \                  |                                   
                          |   \                 |                                   
                          |    \                |                                   
                          |     \_____________  |                                   
                          |                   | |                                   
                          |                   V V                                   
                           \                   xz                                   
                            \                 /    xz = bc*xx + xy              
                             \               /                                      
                              -----> xw <----                                       
                                                                                    
                              xw  = cd*xx + xz                                      


    By the way, you can call 'display_network2()' on an instance of
    ComputationalGraphPrimer to make a much better looking plot of the
    network graph for any DAG created by the sort of expressions shown
    above.

    In the DAG shown above, the variable 'xa' is an independent variable
    since it has no incoming arcs, and 'xw' is an output variable since it
    has no outgoing arcs. A DAG of the sort shown above is represented in
    ComputationalGraphPrimer by two dictionaries: 'depends_on' and 'leads_to'.
    Here is what the 'depends_on' dictionary would look like for the DAG
    shown above:
                                                                                   
        depends_on['xx']  =  ['xa']
        depends_on['xy']  =  ['xa', 'xx']
        depends_on['xz']  =  ['xx', 'xy']
        depends_on['xw']  =  ['xx', 'xz']

    Something like "depends_on['xx'] = ['xa']" is best read as "the vertex
    'xx' depends on the vertex 'xa'."  Similarly, the "depends_on['xz'] =
    ['xx', 'xy']" is best read aloud as "the vertex 'xz' depends on the
    vertices 'xx' and 'xy'." And so on.

    Whereas the 'depends_on' dictionary is a complete description of a DAG,
    for programming convenience, ComputationalGraphPrimer also maintains
    another representation for the same graph, as provide by the 'leads_to'
    dictionary.  This dictionary for the same graph as shown above would
    be:

        leads_to['xa']    =  ['xx', 'xy']
        leads_to['xx']    =  ['xy', 'xz', 'xw']
        leads_to['xy']    =  ['xz']     
        leads_to['xz']    =  ['xw']

     The "leads_to[xa] = [xx]" is best read as "the outgoing edge at the
     node 'xa' leads to the node 'xx'."  Along the same lines, the
     "leads_to['xx'] = ['xy', 'xz', 'xw']" is best read as "the outgoing
     edges at the vertex 'xx' lead to the vertices 'xy', 'xz', and 'xw'.

     Given a computational graph like the one shown above, we are faced
     with the following questions: (1) How to propagate the information
     from the independent nodes --- that we can refer to as the input nodes
     --- to the output nodes, these being the nodes with only incoming
     edges?  (2) As the information flows in the forward direction, meaning
     from the input nodes to the output nodes, is it possible to estimate
     the partial derivatives that apply to each link in the graph?  And,
     finally, (3) Given a scalar value at an output node (which could be
     the loss estimated at that node), can the partial derivatives
     estimated during the forward pass be used to backpropagate the loss?

     Consider, for example, the directed link between the node 'xy' and
     node 'xz'. As a variable, the value of 'xz' is calculated through the
     formula "xz = bc*xx + xy". In the forward propagation of information,
     we estimate the value of 'xz' from currently known values for the
     learnable parameter 'bc' and the variables 'xx' and 'xy'.  In addition
     to the value of the variable at the node 'xz', we are also interested
     in the value of the partial derivative of 'xz' with respect to the
     other variables that it depends on --- 'xx' and 'xy' --- and also with
     respect to the parameter it depends on, 'bc'.  For the calculation of
     the derivatives, we have a choice: We can either do a bit of computer
     algebra and figure out that the partial of 'xz' with respect to 'xx'
     is equal to the current value for 'bc'.  Or, we can use the small
     finite difference method for doing the same, which means that (1) we
     calculate the value of 'xz' for the current value of 'xx', on the one
     hand, and, on the other, for 'xx' plus a delta; (2) take the
     difference of the two; and (3) divide the difference by the delta.
     ComputationalGraphPrimer module uses the finite differences method for
     estimating the partial derivatives.

     Since we have two different types of partial derivatives, partial of a
     variable with respect to another variable, and the partial of a
     variable with respect a learnable parameter, ComputationalGraphPrimer
     uses two different dictionaries for storing this partials during each
     forward pass.  Partials of variables with respect to other variables
     as encountered during forward propagation are stored in the dictionary
     "partial_var_to_var" and the partials of the variables with respect to
     the learnable parameters are stored in the dictionary
     partial_var_to_param.  At the end of each forward pass, the relevant
     partials extracted from these dictionaries are used to estimate the
     gradients of the loss with respect to the learnable parameters, as
     illustrated in the implementation of the method train_on_all_data().

     While the exercise mentioned above is good for appreciating data flows
     in a general DAG, you've got to realize that, with today's algorithms,
     it would be impossible to carry out any learning in a general DAG.  A
     general DAG with millions of learnable parameters would not lend
     itself to a fast calculation of the partial derivatives that are
     needed during the backpropagation step.  Since the exercise described
     above is just to get you thinking about data flows in networks in DAGs
     and nothing else, I have not bothered to include any activation
     functions in the DAG demonstration code in this Primer.

     GOAL 2:

     That brings us to the second major goal of this Primer module:

         To provide examples of simple neural structures in which the
         required partial derivatives are calculated during forward data
         propagation and subsequently used for parameter update during the
         backpropagation of loss.

     In order to become familiar with how this is done in the module, your
     best place to start would be the following two scripts in the Examples
     directory of the distribution:

         one_neuron_classifier.py

         multi_neuron_classifier.py  

     The first script, "one_neuron_classifier.py", invokes the following
     function from the module:

         run_training_loop_one_neuron_model()

     This function, in turn, calls the following functions, the first for
     forward propagation of the data, and the second for the
     backpropagation of loss and updating of the parameters values:

         forward_prop_one_neuron_model()
         backprop_and_update_params_one_neuron_model()

     The data that is forward propagated to the output node is subject to
     Sigmoid activation.  The derivatives that are calculated during
     forward propagation of the data include the partial 'output vs. input'
     derivatives for the Sigmoid nonlinearity. The backpropagation step
     implemented in the second of the two functions listed above includes
     averaging the partial derivatives and the prediction errors over a
     batch of training samples, as required by SGD.
     
     The second demo script in the Examples directory,
     "multi_neuron_classifier.py" creates a neural network with a hidden
     layer and an output layer.  Each node in the hidden layer and the node
     in the output layer are all subject to Sigmoid activation.
     This script invokes the following function of the module:

         run_training_loop_multi_neuron_model()

     And this function, in turn, calls upon the following two functions,
     the first for forward propagating the data and the second for the
     backpropagation of loss and updating of the parameters:

        forward_prop_multi_neuron_model()
        backprop_and_update_params_multi_neuron_model()

     In contrast with the one-neuron demo, in this case, the batch-based
     data that is output by the forward function is sent directly to the
     backprop function.  It then becomes the job of the backprop function
     to do the averaging needed for SGD.

     In the Examples directory, you will also find the following script:

        verify_with_torchnn.py

     The idea for this script is to serve as a check on the performance of
     the main demo scripts "one_neuron_classifier.py" and
     "multi_neuron_classifier.py".  Note that you cannot expect the
     performance of my one-neuron and multi-neuron scripts to match what
     you would get from similar networks constructed with components drawn
     from "torch.nn".  One primary reason for that is that "torch.nn" based
     code uses the state-of-the-art optimization of the steps in the
     parameter hyperplane, with is not the case with my demo scripts.
     Nonetheless, a comparison with the "torch.nn" is important for general
     trend of how the training loss varies with the iterations.  That is,
     if the "torch.nn" based script showed decreasing loss (indicated that
     learning was taking place) while that was not the case with my
     one-neuron and multi-neuron scripts, that would indicate that perhaps
     I had made an error in either the computation of the partial derivatives
     during the forward propagation of the data, or I had used the
     derivatives for updating the parameters.

     GOAL 3:

     The goal here is to show how to extend PyTorch's Autograd class if you
     want to endow it with additional functionality. Let's say that you
     wish for some data condition to be remembered during the forward
     propagation of the data through a network and then use that condition
     to alter in some manner how the parameters would be updated during
     backpropagation of the prediction errors.  This can be accomplished by
     subclassing from Autograd and incorporating the desired behavior in
     the subclass.  As to how how you can extend Autograd is demonstrated
     by the inner class AutogradCustomization in this module. Your starting
     point for understanding what this class does would be the script

         extending_autograd.py

     in the Examples directory of the distribution. 


@title
INSTALLATION:

    The ComputationalGraphPrimer class was packaged using setuptools.  For
    installation, execute the following command in the source directory
    (this is the directory that contains the setup.py file after you have
    downloaded and uncompressed the package):
 
            sudo python3 setup.py install

    On Linux distributions, this will install the module file at a location
    that looks like

             /usr/local/lib/python3.8/dist-packages/

    If you do not have root access, you have the option of working directly
    off the directory in which you downloaded the software by simply
    placing the following statements at the top of your scripts that use
    the ComputationalGraphPrimer class:

            import sys
            sys.path.append( "pathname_to_ComputationalGraphPrimer_directory" )

    To uninstall the module, simply delete the source directory, locate
    where the ComputationalGraphPrimer module was installed with "locate
    ComputationalGraphPrimer" and delete those files.  As mentioned above,
    the full pathname to the installed version is likely to look like
    "/usr/local/lib/python3.8/dist-packages/".

    If you want to carry out a non-standard install of the
    ComputationalGraphPrimer module, look up the on-line information on
    Disutils by pointing your browser to

              http://docs.python.org/dist/dist.html

@title
USAGE:

    Construct an instance of the ComputationalGraphPrimer class as follows:

        from ComputationalGraphPrimer import *

        cgp = ComputationalGraphPrimer(
                       expressions = ['xx=xa^2',
                                      'xy=ab*xx+ac*xa',
                                      'xz=bc*xx+xy',
                                      'xw=cd*xx+xz^3'],
                       output_vars = ['xw'],
                       dataset_size = 10000,
                       learning_rate = 1e-6,
                       grad_delta    = 1e-4,
                       display_loss_how_often =	1000,
              )
        
        cgp.parse_expressions()
        cgp.display_network2()                                                                    
        cgp.gen_gt_dataset(vals_for_learnable_params = {'ab':1.0, 'bc':2.0, 'cd':3.0, 'ac':4.0})
        cgp.train_on_all_data()
        cgp.plot_loss()


@title
CONSTRUCTOR PARAMETERS: 

    batch_size: Introduced in Version 1.0.5 for demonstrating forward
                    propagation of the input data while calculating the
                    partial derivatives needed during backpropagation of
                    loss. For SGD, updating the parameters involves
                    smoothing the derivatives over the training samples in
                    a batch. Hence the need for batch_size as a constructor
                    parameter.

    dataset_size: Although the networks created by an arbitrary set of
                    expressions are not likely to allow for any true
                    learning of the parameters, nonetheless the
                    ComputationalGraphPrimer allows for the computation of
                    the loss at the output nodes and backpropagation of the
                    loss to the other nodes.  To demonstrate this, we need
                    a ground-truth set of input/output values for given
                    value for the learnable parameters.  The constructor
                    parameter 'dataset_size' refers to how may of these
                    'input/output' pairs would be generated for such
                    experiments.

                    For the one-neuron and multi-neuron demos introduced in
                    Version 1.0.5, the constructor parameter dataset_size
                    refers to many tuples of randomly generated data should
                    be made available for learning. The size of each data
                    tuple is deduced from the the first expression in the
                    list made available to module through the parameter
                    'expressions' described below.

    display_loss_how_often: This controls how often you will see the result
                    of the calculations being carried out in the
                    computational graph.  Let's say you are experimenting
                    with 10,000 input/output samples for propagation in the
                    network, if you set this constructor option to 1000,
                    you will see the partial derivatives and the values for
                    the learnable parameters every 1000 passes through the
                    graph.

    expressions: These expressions define the computational graph.  The
                    expressions are based on the following assumptions: (1)
                    any variable name must start with the letter 'x'; (2) a
                    symbolic name that does not start with 'x' is a
                    learnable parameter; (3) exponentiation operator is
                    '^'; (4) the symbols '*', '+', and '-' carry their
                    usual arithmetic meanings.

    grad_delta: This constructor option sets the value of the delta to be
                    used for estimating the partial derivatives with the
                    finite difference method.

    layers_config: Introduced in Version 1.0.5 for the multi-neuron
                    demo. Its value is a list of nodes in each layer of the
                    network. Note that I consider the input to the neural
                    network as a layer unto itself.  Therefore, if the
                    value of the parameter num_layers is 3, the list you
                    supply for layers_config must have three numbers in it.

    learning_rate: Carries the usual meaning for updating the values of the
                    learnable parameters based on the gradients of the loss
                    with respect to those parameters.

    num_layers: Introduced in Version 1.0.5 for the multi-neuron demo. It
                    is merely a convenience parameter that indicated the
                    number of layers in your multi-neuron network. For the
                    purpose of counting layers, I consider the input as a
                    layer unto itself.

    one_neuron_model: Introduced in Version 1.0.5.  This boolean parameter
                    is needed only when you are constructing a one-neuron
                    demo. I needed this constructor parameter for some
                    conditional evaluations in the "parse_expressions()"
                    method of the module.  I use that expression parser for
                    both the older demos and the new demo based on the
                    one-neuron model.

    output_vars: Although the parser has the ability to figure out which
                    nodes in the computational graph represent the output
                    variables --- these being nodes with no outgoing arcs
                    --- you are allowed to designate the specific output
                    variables you are interested in through this
                    constructor parameter.

    training_iterations: Carries the expected meaning.


@title
PUBLIC METHODS:

    (1)  backprop_and_update_params_one_neuron_model():

         Introduced in Version 1.0.5.  This method is called by
         run_training_loop_one_neuron_model() for backpropagating the loss
         and updating the values of the learnable parameters.

    (2)  backprop_and_update_params_multi_neuron_model():

         Introduced in Version 1.0.5.  This method is called by
         run_training_loop_multi_neuron_model() for backpropagating the
         loss and updating the values of the learnable parameters.

    (3)  display_network2():

         This method calls on the networkx module to construct a visual
         display of the computational graph.


    (4)  forward_propagate_one_input_sample_with_partial_deriv_calc():

         This method is used for pushing the input data forward through a
         general DAG and at the same computing the partial derivatives that
         would be needed during backpropagation for updating the values of
         the learnable parameters.

    (5)  forward_prop_one_neuron_model():

         Introduced in Version 1.0.5.  This function propagates the input
         data through a one-neuron network.  The data aggregated at the
         neuron is subject to a Sigmoid activation.  The function also
         calculates the partial derivatives needed during backprop.

    (6)  forward_prop_multi_neuron_model():

         Introduced in Version 1.0.5. This function does the same thing as
         the previous function, except that it is intended for a multi-layer
         neural network. The pre-activation values at each neuron are
         subject to the Sigmoid nonlinearity. At the same time, the partial
         derivatives are calculated and stored away for use during backprop.

    (7)  gen_gt_dataset()

         This method generates the training data for a general graph of
         nodes in a DAG. For random values at the input nodes, it
         calculates the values at the output nodes assuming certain given
         values for the learnable parameters in the network. If it were
         possible to carry out learning in such a network, the goal would
         to see if the value of those parameters would be learned
         automatically as in a neural network.

    (8)  gen_training_data():

         Introduced in Version 1.0.5. This function generates training data
         for the scripts "one_neuron_classifier.py",
         "multi_neuron_classifier.py" and "verify_with_torchnn.py" scripts
         in the Examples directory of the distribution.  The data
         corresponds to two classes defined by two different multi-variate
         distributions. The dimensionality of the data is determined
         entirely the how many nodes are found by the expression parser in
         the list of expressions that define the network.

    (9)  parse_expressions()

         This method parses the expressions provided and constructs a DAG
         from them for the variables and the parameters in the expressions.
         It is based on the convention that the names of all variables
         begin with the character 'x', with all other symbolic names being
         treated as learnable parameters.

    (10) parse_multi_layer_expressions():

         Introduced in Version 1.0.5. Whereas the previous method,
         parse_expressions(), works well for creating a general DAG and for
         the one-neuron model, it is not meant to capture the layer based
         structure of a neural network.  Hence this method.

    (11) run_training_loop_one_neuron_model():

         Introduced in Version 1.0.5.  This is the main function in the
         module for the demo based on the one-neuron model. The demo
         consists of propagating the input values forward, aggregating them
         at the neuron, and subjecting the result to Sigmoid activation.
         All the partial derivatives needed for updating the link weights
         are calculating the forward propagation.  This includes the
         derivatives of the output vis-a-vis the input at the Sigmoid
         activation.  Subsequently, during backpropagation of the loss, the
         parameter values are updated using the derivatives stored away
         during forward propagation.

    (12) run_training_loop_multi_neuron_model()

         Introduced in Version 1.0.5.  This is the main function for the
         demo based on a multi-layer neural network.  As each batch of
         training data is pushed through the network, the partial derivatives
         of the output at each layer is computed with respect to the
         parameters. This calculating includes computing the partial
         derivatives at the output of the activation function with respect
         to its input.  Subsequently, during backpropagation, first
         batch-based smoothing is applied to the derivatives and the
         prediction errors stored away during forward propagation in order
         to comply with the needs of SGD and the values of the learnable
         parameters updated.
         
    (13) run_training_with_torchnn():

         Introduced in Version 1.0.5.  The purpose of this function is to
         use comparable network components from the torch.nn module in
         order to "authenticate" the performance of the handcrafted
         one-neuron and the multi-neuron models in this module.  All that
         is meant by "authentication" here is that if the torch.nn based
         networks show the training loss decrease with iterations, you
         would the one-neuron and the multi-neuron models to show similar
         results.  This function contains the following inner classes:

                  class OneNeuronNet( torch.nn.Module )

                  class MultiNeuronNet( torch.nn.Module )

         that define networks similar to the handcrafted one-neuron and
         multi-neuron networks of this module.


    (14) train_on_all_data()

         The purpose of this function is to call forward propagation and
         backpropagation functions of the module for the demo based on
         arbitrary DAGs.


    (15) plot_loss()

         This is only used by the functions that DAG based demonstration code
         in the module.  The training functions introduced in Version 1.0.5 have
         embedded code for plotting the loss as a function of iterations.


@title 
THE Examples DIRECTORY:

    The Examples directory of the distribution contains the following the
    following scripts:

    1.   graph_based_dataflow.py

             This demonstrates forward propagation of input data and
             backpropagation in a general DAG (Directed Acyclic Graph).
             The forward propagation involves estimating the partial
             derivatives that would subsequently be used for "updating" the
             learnable parameters during backpropagation.  Since I have not
             incorporated any activations in the DAG, you can really not
             expect any real learning to take place in this demo.  The
             purpose of this demo is just to illustrate what is meant by a
             DAG and how information can flow forwards and backwards in
             such a network.

    2.   one_neuron_classifier.py

             This script demonstrates the one-neuron model in the module.
             The goal is to show forward propagation of data through the
             neuron (which includes the Sigmoid activation), while
             calculating the partial derivatives needed during the
             backpropagation step for updating the parameters.

    3.   multi_neuron_classifier.py  

             This script generalizes what is demonstrated by the one-neuron
             model to a multi-layer neural network.  This script
             demonstrates saving the partial-derivative information
             calculated during the forward propagation through a
             multi-layer neural network and using that information for
             backpropagating the loss and for updating the values of the
             learnable parameters.

    4.   verify_with_torchnn.py

              The purpose of this script is just to verify that the results
              obtained with the scripts "one_neuron_classifier.py" and
              "multi_neuron_classifier.py" are along the expected lines.
              That is, if similar networks constructed with the torch.nn
              module show the training loss decreasing with iterations, you
              would expect the similar learning behavior from the scripts
              "one_neuron_classifier.py" and "multi_neuron_classifier.py".

    5.  extending_autograd.py

              This provides a demo example of the recommended approach for
              giving additional functionality to Autograd.  See the
              explanation in the doc section associated with the inner
              class AutogradCustomization of this module for further info.


@title
BUGS:

    Please notify the author if you encounter any bugs.  When sending
    email, please place the string 'ComputationalGraphPrimer' in the
    subject line to get past the author's spam filter.


@title
ABOUT THE AUTHOR:

    The author, Avinash Kak, is a professor of Electrical and Computer
    Engineering at Purdue University.  For all issues related to this
    module, contact the author at kak@purdue.edu If you send email, please
    place the string "ComputationalGraphPrimer" in your subject line to get
    past the author's spam filter.

@title
COPYRIGHT:

    Python Software Foundation License

    Copyright 2021 Avinash Kak

@endofdocs
'''


import sys,os,os.path
import numpy as np
import re
import operator
import math
import random
import torch
from collections import deque
import copy
import matplotlib.pyplot as plt
import networkx as nx


class Exp:
    def __init__(self, exp, body, dependent_var, right_vars, right_params):
        self.exp = exp
        self.body = body
        self.dependent_var = dependent_var
        self.right_vars = right_vars
        self.right_params = right_params

#______________________________  ComputationalGraphPrimer Class Definition  ________________________________

class ComputationalGraphPrimer(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''ComputationalGraphPrimer constructor can only be called with keyword arguments for 
                      the following keywords: expressions, output_vars, dataset_size, grad_delta,
                      learning_rate, display_loss_how_often, one_neuron_model, training_iterations, 
                      batch_size, num_layers, layers_config, epochs, and debug''')
        expressions = output_vars = dataset_size = grad_delta = display_loss_how_often = learning_rate = one_neuron_model = training_iterations = batch_size = num_layers = layers_config = epochs = debug  = None
        if 'one_neuron_model' in kwargs              :   one_neuron_model = kwargs.pop('one_neuron_model')
        if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
        if 'num_layers' in kwargs                    :   num_layers = kwargs.pop('num_layers')
        if 'layers_config' in kwargs                 :   layers_config = kwargs.pop('layers_config')
        if 'expressions' in kwargs                   :   expressions = kwargs.pop('expressions')
        if 'output_vars' in kwargs                   :   output_vars = kwargs.pop('output_vars')
        if 'dataset_size' in kwargs                  :   dataset_size = kwargs.pop('dataset_size')
        if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
        if 'training_iterations' in kwargs           :   training_iterations = \
                                                                   kwargs.pop('training_iterations')
        if 'grad_delta' in kwargs                    :   grad_delta = kwargs.pop('grad_delta')
        if 'display_loss_how_often' in kwargs        :   display_loss_how_often = kwargs.pop('display_loss_how_often')
        if 'epochs' in kwargs                        :   epochs = kwargs.pop('epochs')
        if 'debug' in kwargs                         :   debug = kwargs.pop('debug') 
        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        self.one_neuron_model =  True if one_neuron_model is not None else False
        if training_iterations:
            self.training_iterations = training_iterations
        self.batch_size  =  batch_size if batch_size else 4
        self.num_layers = num_layers 
        if layers_config:
            self.layers_config = layers_config
        if expressions:
            self.expressions = expressions
#        else:
#            sys.exit("you need to supply a list of expressions")
        if output_vars:
            self.output_vars = output_vars
        if dataset_size:
            self.dataset_size = dataset_size
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-6
        if grad_delta:
            self.grad_delta = grad_delta
        else:
            self.grad_delta = 1e-4
        if display_loss_how_often:
            self.display_loss_how_often = display_loss_how_often
        if dataset_size:
            self.dataset_input_samples  = {i : None for i in range(dataset_size)}
            self.true_output_vals       = {i : None for i in range(dataset_size)}
        self.vals_for_learnable_params = None
        self.epochs = epochs
        if debug:                             
            self.debug = debug
        else:
            self.debug = 0
        self.independent_vars = None
        self.gradient_of_loss = None
        self.gradients_for_learnable_params = None
        self.expressions_dict = {}
        self.LOSS = []                               ##  loss values for all iterations of training
        self.all_vars = set()
        if (one_neuron_model is True) or (num_layers is not None):
            self.independent_vars = []
            self.learnable_params = []
        else:
            self.independent_vars = set()
            self.learnable_params = set()
        self.dependent_vars = {}
        self.depends_on = {}                         ##  See Introduction for the meaning of this 
        self.leads_to = {}                           ##  See Introduction for the meaning of this 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    def parse_expressions(self):
        ''' 
        This method creates a DAG from a set of expressions that involve variables and learnable
        parameters. The expressions are based on the assumption that a symbolic name that starts
        with the letter 'x' is a variable, with all other symbolic names being learnable parameters.
        The computational graph is represented by two dictionaries, 'depends_on' and 'leads_to'.
        To illustrate the meaning of the dictionaries, something like "depends_on['xz']" would be
        set to a list of all other variables whose outgoing arcs end in the node 'xz'.  So 
        something like "depends_on['xz']" is best read as "node 'xz' depends on ...." where the
        dots stand for the array of nodes that is the value of "depends_on['xz']".  On the other
        hand, the 'leads_to' dictionary has the opposite meaning.  That is, something like
        "leads_to['xz']" is set to the array of nodes at the ends of all the arcs that emanate
        from 'xz'.
        '''
        self.exp_objects = []
        for exp in self.expressions:
            left,right = exp.split('=')
            self.all_vars.add(left)
            self.expressions_dict[left] = right
            self.depends_on[left] = []
            parts = re.findall('([a-zA-Z]+)', right)
            right_vars = []
            right_params = []
            for part in parts:
                if part.startswith('x'):
                    self.all_vars.add(part)
                    self.depends_on[left].append(part)
                    right_vars.append(part)
                else:
                    if self.one_neuron_model is True:
                        self.learnable_params.append(part)
                    else:
                        self.learnable_params.add(part)
                    right_params.append(part)
            exp_obj = Exp(exp, right, left, right_vars, right_params)
            self.exp_objects.append(exp_obj)
        if self.debug:
            print("\n\nall variables: %s" % str(self.all_vars))
            print("\n\nlearnable params: %s" % str(self.learnable_params))
            print("\n\ndependencies: %s" % str(self.depends_on))
            print("\n\nexpressions dict: %s" % str(self.expressions_dict))
        for var in self.all_vars:
            if var not in self.depends_on:              # that is, var is not a key in the depends_on dict
                if self.one_neuron_model is True:
                    self.independent_vars.append(var)                
                else:
                    self.independent_vars.add(var)
        self.input_size = len(self.independent_vars)
        if self.debug:
            print("\n\nindependent vars: %s" % str(self.independent_vars))
        self.dependent_vars = [var for var in self.all_vars if var not in self.independent_vars]
        self.output_size = len(self.dependent_vars)
        self.leads_to = {var : set() for var in self.all_vars}
        for k,v in self.depends_on.items():
            for var in v:
                self.leads_to[var].add(k)    
        if self.debug:
            print("\n\nleads_to dictionary: %s" % str(self.leads_to))


    def parse_multi_layer_expressions(self):
        ''' 
        This method is a modification of the previous expression parser and meant specifically
        for the case when a given set of expressions are supposed to define a multi-layer neural
        network.  The naming conventions for the variables, which designate  the nodes in the layers
        of the network, and the learnable parameters remain the same as in the previous function.
        '''
        self.exp_objects = []
        self.layer_expressions = { i : [] for i in range(1,self.num_layers) }
        self.layer_exp_objects = { i : [] for i in range(1,self.num_layers) }
        all_expressions = deque(self.expressions)
        for layer_index in range(self.num_layers - 1):
            for node_index in range(self.layers_config[layer_index+1]):   
                self.layer_expressions[layer_index+1].append( all_expressions.popleft() )
        print("\n\nself.layer_expressions: ", self.layer_expressions)
        self.layer_vars = {i : [] for i in range(self.num_layers)}         # layer indexing starts at 0
        self.layer_params = {i : [] for i in range(1,self.num_layers)}     # layer indexing starts at 1
        for layer_index in range(1,self.num_layers):
            for exp in self.layer_expressions[layer_index]:
                left,right = exp.split('=')
                self.all_vars.add(left)
                self.expressions_dict[left] = right
                self.depends_on[left] = []
                parts = re.findall('([a-zA-Z]+)', right)
                right_vars = []
                right_params = []
                for part in parts:
                    if part.startswith('x'):
                        self.all_vars.add(part)
                        self.depends_on[left].append(part)
                        right_vars.append(part)
                    else:
                        if (self.one_neuron_model is True) or (self.num_layers is not None):
                            self.learnable_params.append(part)
                        else:
                            self.learnable_params.add(part)
                        right_params.append(part)
                self.layer_vars[layer_index-1] = right_vars
                self.layer_vars[layer_index].append(left)
                self.layer_params[layer_index].append(right_params)
                exp_obj = Exp(exp, right, left, right_vars, right_params)
                ##  when num_layers is defined and >0, the sequence of expression in 
                ##  self.exp_objects would correspond to layers
                self.layer_exp_objects[layer_index].append(exp_obj)
            if self.debug:
                print("\n\nall variables: %s" % str(self.all_vars))
                print("\n\nlearnable params: %s" % str(self.learnable_params))
                print("\n\ndependencies: %s" % str(self.depends_on))
                print("\n\nexpressions dict: %s" % str(self.expressions_dict))

            for var in self.all_vars:
                if var not in self.depends_on:              # that is, var is not a key in the depends_on dict
                    if (self.one_neuron_model is True) or (self.num_layers is not None):
                        self.independent_vars.append(var)                
                    else:
                        self.independent_vars.add(var)
            self.input_size = len(self.independent_vars)
            if self.debug:
                print("\n\nindependent vars: %s" % str(self.independent_vars))
            self.dependent_vars = [var for var in self.all_vars if var not in self.independent_vars]
            self.output_size = len(self.dependent_vars)
            self.leads_to = {var : set() for var in self.all_vars}
            for k,v in self.depends_on.items():
                for var in v:
                    self.leads_to[var].add(k)    
            if self.debug:
                print("\n\nleads_to dictionary: %s" % str(self.leads_to))
        print("\n\nself.layer_vars: ", self.layer_vars)
        print("\n\nself.layer_params: ", self.layer_params)
        print("\n\nself.layer_exp_objects: ", self.layer_exp_objects)



    def display_network1(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.all_vars)
        edges = []
        for ver1 in self.leads_to:
            for ver2 in self.leads_to[ver1]:
                edges.append( (ver1,ver2) )
        G.add_edges_from( edges )
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

    def display_network2(self):
        '''
        Provides a fancier display of the network graph
        '''
        G = nx.DiGraph()
        G.add_nodes_from(self.all_vars)
        edges = []
        for ver1 in self.leads_to:
            for ver2 in self.leads_to[ver1]:
                edges.append( (ver1,ver2) )
        G.add_edges_from( edges )
        pos = nx.circular_layout(G)    
        nx.draw(G, pos, with_labels = True, edge_color='b', node_color='lightgray', 
                          arrowsize=20, arrowstyle='fancy', node_size=1200, font_size=20, 
                          font_color='black')
        plt.title("Computational graph for the expressions")
        plt.show()



    ### Introduced in 1.0.5
    ######################################################################################################
    ######################################### one neuron model ###########################################
    def run_training_loop_one_neuron_model(self, mu):
        training_data = self.training_data
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        self.bias = random.uniform(0,1)
        self.mu = mu
        self.velocity = 0
        self.bias_step = 0
        
        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]
            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])
            def _getitem(self):    
                cointoss = random.choice([0,1])
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            
            def getbatch(self):
                batch_data,batch_labels = [],[]
                maxval = 0.0
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]                
                batch = [batch_data, batch_labels]
                return batch                

        data_loader = DataLoader(self.training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_literations = 0.0
        
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])
            loss_avg = loss / float(len(class_labels))
            avg_loss_over_literations += loss_avg
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_literations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_literations)
                #print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_literations))
                avg_loss_over_literations = 0.0
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
        # plt.figure()     
        # plt.plot(loss_running_record) 
        return loss_running_record
        
    def forward_prop_one_neuron_model(self, data_tuples_in_batch):
        """
        As the one-neuron model is characterized by a single expression, the main job of this function is
        to evaluate that expression for each data tuple in the incoming batch.  The resulting output is
        fed into the sigmoid activation function and the partial derivative of the sigmoid with respect
        to its input calcualated.
        """
        output_vals = []
        deriv_sigmoids = []
        for vals_for_input_vars in data_tuples_in_batch:
            input_vars = self.independent_vars
            vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
            exp_obj = self.exp_objects[0]
            output_val = self.eval_expression(exp_obj.body , vals_for_input_vars_dict, self.vals_for_learnable_params)
            output_val = output_val + self.bias
            ## apply sigmoid activation:
            output_val = 1.0 / (1.0 + np.exp(-1.0 * output_val))
            ## calculate partial of the activation function as a function of its input
            deriv_sigmoid = output_val * (1.0 - output_val)
            output_vals.append(output_val)
            deriv_sigmoids.append(deriv_sigmoid)
        return output_vals, deriv_sigmoids

    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        As should be evident from the syntax used in the following call to backprop function,

           self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                     ^^^             ^^^                ^^^
                                                               
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.
        """
        input_vars = self.independent_vars
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            step = self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid + self.mu * self.velocity
            self.vals_for_learnable_params[param] += step
        self.velocity = step
        self.bias_step = self.learning_rate * y_error * deriv_sigmoid + self.mu * self.bias_step
        self.bias += self.bias_step
    ######################################################################################################


    ### Introduced in 1.0.5
    ######################################################################################################
    ######################################## multi neuron model ##########################################
    def run_training_loop_multi_neuron_model(self, mu):
        training_data = self.training_data
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        self.bias = [random.uniform(0,1) for _ in range(self.num_layers-1)]
        self.mu = mu
        self.velocity = 0
        self.bias_step = [0, 0]
        
        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]
            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])
            def _getitem(self):    
                cointoss = random.choice([0,1])
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            
            def getbatch(self):
                batch_data,batch_labels = [],[]
                maxval = 0.0
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]                
                batch = [batch_data, batch_labels]
                return batch                

        data_loader = DataLoader(self.training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_literations = 0.0
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)
            predicted_labels_for_batch = self.forw_prop_vals_at_layers[self.num_layers-1]
            y_preds =  [item for sublist in  predicted_labels_for_batch  for item in sublist]
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])
            loss_avg = loss / float(len(class_labels))
            avg_loss_over_literations += loss_avg
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_literations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_literations)
                # print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_literations))
                avg_loss_over_literations = 0.0
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            self.backprop_and_update_params_multi_neuron_model(y_error_avg, class_labels)
        return loss_running_record


    def forward_prop_multi_neuron_model(self, data_tuples_in_batch):
        """
        During forward propagation, we push each batch of the input data through the
        network.  In order to explain the logic of forward, consider the following network
        layout in 4 nodes in the input layer, 2 nodes in the hidden layer, and 1 node in
        the output layer.

                               input
                                  
                                 x                                             x = node
                                                                            
                                 x         x|                                  | = sigmoid activation
                                                     x|
                                 x         x|   

                                 x
                            
                             layer_0    layer_1    layer_2

                
        In the code shown below, the expressions to evaluate for computing the
        pre-activation values at a node are stored at the layer in which the nodes reside.
        That is, the dictionary look-up "self.layer_exp_objects[layer_index]" returns the
        Expression objects for which the left-side dependent variable is in the layer
        pointed to layer_index.  So the example shown above, "self.layer_exp_objects[1]"
        will return two Expression objects, one for each of the two nodes in the second
        layer of the network (that is, layer indexed 1).

        The pre-activation values obtained by evaluating the expressions at each node are
        then subject to Sigmoid activation, followed by the calculation of the partial
        derivative of the output of the Sigmoid function with respect to its input.

        In the forward, the values calculated for the nodes in each layer are stored in
        the dictionary

                        self.forw_prop_vals_at_layers[ layer_index ]

        and the gradients values calculated at the same nodes in the dictionary:

                        self.gradient_vals_for_layers[ layer_index ]

        """
        self.forw_prop_vals_at_layers = {i : [] for i in range(self.num_layers)}   
        self.gradient_vals_for_layers = {i : [] for i in range(1, self.num_layers)}
        for vals_for_input_vars in data_tuples_in_batch:
            self.forw_prop_vals_at_layers[0].append(vals_for_input_vars)
            for layer_index in range(1, self.num_layers):
                input_vars = self.layer_vars[layer_index-1]
                if layer_index == 1:
                    vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
                output_vals_arr = []
                gradients_val_arr = []
                for exp_obj in self.layer_exp_objects[layer_index]:
                    output_val = self.eval_expression(exp_obj.body , vals_for_input_vars_dict,    
                                                                 self.vals_for_learnable_params, input_vars)
                    output_val = output_val + self.bias[layer_index-1]                
                    ## apply sigmoid activation:
                    output_val = 1.0 / (1.0 + np.exp(-1.0 * output_val))
                    output_vals_arr.append(output_val)
                    ## calculate partial of the activation function as a function of its input
                    deriv_sigmoid = output_val * (1.0 - output_val)
                    gradients_val_arr.append(deriv_sigmoid)
                    vals_for_input_vars_dict[ exp_obj.dependent_var ] = output_val
                self.forw_prop_vals_at_layers[layer_index].append(output_vals_arr)
                self.gradient_vals_for_layers[layer_index].append(gradients_val_arr)

    def backprop_and_update_params_multi_neuron_model(self, y_error, class_labels):
        """
        First note that loop index variable 'back_layer_index' starts with the index of
        the last layer.  For the 3-layer example shown for 'forward', back_layer_index
        starts with a value of 2, its next value is 1, and that's it.

        Stochastic Gradient Gradient calls for the backpropagated loss to be averaged over
        the samples in a batch.  To explain how this averaging is carried out by the
        backprop function, consider the last node on the example shown in the forward()
        function above.  Standing at the node, we look at the 'input' values stored in the
        variable "input_vals".  Assuming a batch size of 8, this will be list of
        lists. Each of the inner lists will have two values for the two nodes in the
        hidden layer. And there will be 8 of these for the 8 elements of the batch.  We average
        these values 'input vals' and store those in the variable "input_vals_avg".  Next we
        must carry out the same batch-based averaging for the partial derivatives stored in the
        variable "deriv_sigmoid".

        Pay attention to the variable 'vars_in_layer'.  These stores the node variables in
        the current layer during backpropagation.  Since back_layer_index starts with a
        value of 2, the variable 'vars_in_layer' will have just the single node for the
        example shown for forward(). With respect to what is stored in vars_in_layer', the
        variables stored in 'input_vars_to_layer' correspond to the input layer with
        respect to the current layer. 
        """
        # backproped prediction error:
        pred_err_backproped_at_layers = {i : [] for i in range(1,self.num_layers-1)}  
        pred_err_backproped_at_layers[self.num_layers-1] = [y_error]
        for back_layer_index in reversed(range(1,self.num_layers)):
            input_vals = self.forw_prop_vals_at_layers[back_layer_index -1]
            input_vals_avg = [sum(x) for x in zip(*input_vals)]
            input_vals_avg = list(map(operator.truediv, input_vals_avg, [float(len(class_labels))] * len(class_labels)))
            deriv_sigmoid =  self.gradient_vals_for_layers[back_layer_index]
            deriv_sigmoid_avg = [sum(x) for x in zip(*deriv_sigmoid)]
            deriv_sigmoid_avg = list(map(operator.truediv, deriv_sigmoid_avg, 
                                                             [float(len(class_labels))] * len(class_labels)))
            vars_in_layer  =  self.layer_vars[back_layer_index]                 ## a list like ['xo']
            vars_in_next_layer_back  =  self.layer_vars[back_layer_index - 1]   ## a list like ['xw', 'xz']

            layer_params = self.layer_params[back_layer_index]         
            ## note that layer_params are stored in a dict like        
                ##     {1: [['ap', 'aq', 'ar', 'as'], ['bp', 'bq', 'br', 'bs']], 2: [['cp', 'cq']]}
            ## "layer_params[idx]" is a list of lists for the link weights in layer whose output nodes are in layer "idx"
            transposed_layer_params = list(zip(*layer_params))         ## creating a transpose of the link matrix

            backproped_error = [None] * len(vars_in_next_layer_back)
            for k,varr in enumerate(vars_in_next_layer_back):
                for j,var2 in enumerate(vars_in_layer):
                    backproped_error[k] = sum([self.vals_for_learnable_params[transposed_layer_params[k][i]] * 
                                               pred_err_backproped_at_layers[back_layer_index][i] 
                                               for i in range(len(vars_in_layer))])
#                                               deriv_sigmoid_avg[i] for i in range(len(vars_in_layer))])
            pred_err_backproped_at_layers[back_layer_index - 1]  =  backproped_error
            input_vars_to_layer = self.layer_vars[back_layer_index-1]
            for j,var in enumerate(vars_in_layer):
                layer_params = self.layer_params[back_layer_index][j]
                for i,param in enumerate(layer_params):
                    gradient_of_loss_for_param = input_vals_avg[i] * pred_err_backproped_at_layers[back_layer_index][j] 
                    step = self.learning_rate * gradient_of_loss_for_param * deriv_sigmoid_avg[j] + self.mu * self.velocity
                    self.vals_for_learnable_params[param] += step
                self.velocity = step   
        self.bias_step[back_layer_index-1] = self.learning_rate * sum(pred_err_backproped_at_layers[back_layer_index]) * sum(deriv_sigmoid_avg)/len(deriv_sigmoid_avg) + self.mu * self.bias_step[back_layer_index-1] 
        self.bias[back_layer_index-1] += self.bias_step[back_layer_index-1]
        
    ######################################################################################################


    ### Introduced in 1.0.5
    ######################################################################################################
    ############################# torch.nn based experiments for verification ############################
    def run_training_with_torchnn(self, option):
        """
        The value of the parameter 'option' must be either 'one_neuron' or 'multi_neuron'.

        For either option, the number of input nodes is specified by the expressions specified in the        
        contructor of the class ComputationalGraphPrimer.

        When the option value is 'one_neuron', we use the OneNeuronNet for the learning network and
        when the option is 'multi_neuron' we use the MultiNeuronNet.

        Assuming that the number of input nodes specified by the expressions is 4, the MultiNeuronNet 
        class creates the following network layout in which we have 2 nodes in the hidden layer and 
        one node for the final output:

                               input
                                  
                                 x                                             x = node
                                                                            
                                 x         x|                                  | = ReLU activation
                                                     x|
                                 x         x|   

                                 x
                            
                             layer_0    layer_1    layer_2


        """
        training_data = self.training_data

        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]
            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])
            def _getitem(self):    
                cointoss = random.choice([0,1])
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            
            def getbatch(self):
                batch_data,batch_labels = [],[]
                maxval = 0.0
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]                
                batch = [batch_data, batch_labels]
                return batch                

        data_loader = DataLoader(self.training_data, batch_size=self.batch_size)

        class OneNeuronNet(torch.nn.Module):
            """
            This class is used when the parameter 'option' is set to 'one_neuron' in the call to
            this training function.
            """
            def __init__(self, D_in, D_out):
                torch.nn.Module.__init__( self )
                self.linear = torch.nn.Linear(D_in, D_out)
                self.sigmoid = torch.nn.Sigmoid()
            def forward(self, x):
                h_out = self.linear(x)
                y_pred = self.sigmoid(h_out)
                return y_pred

        class MultiNeuronNet(torch.nn.Module):
            """
            This class is used when the parameter 'option' is set to 'multi_neuron' in the call to
            this training function.
            """
            def __init__(self, D_in, H, D_out):
                torch.nn.Module.__init__( self )
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)
            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred

        loss_running_record = []
        i = 0
        avg_loss_over_literations = 0.0
        if option == 'one_neuron':
            N,D_in,D_out = self.batch_size,self.input_size,self.output_size
            model = OneNeuronNet(D_in,D_out)
        elif option == 'multi_neuron':
            N,D_in,H,D_out = self.batch_size,self.input_size,2,self.output_size
            model = MultiNeuronNet(D_in,H,D_out)
        else:
            sys.exit("\n\nThe value of the parameter 'option' not recognized\n\n")
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), self.learning_rate)
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = torch.FloatTensor( data[0] )
            class_labels = torch.FloatTensor( data[1] )
            # We need to convert the shape torch.Size([8]) into the shape torch.Size([8, 1]):
            class_labels = torch.unsqueeze(class_labels, 1)    
            y_preds = model(data_tuples)
            loss = criterion(y_preds, class_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss_over_literations += loss.item()
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_literations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_literations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_literations))
                avg_loss_over_literations = 0.0
        plt.figure()     
        plt.plot(loss_running_record) 
        plt.show()   
    ######################################################################################################


    ######################################################################################################
    ###############################    for a general graph of nodes   ####################################
    def train_on_all_data(self):
        '''
        The purpose of this method is to call forward_propagate_one_input_sample_with_partial_deriv_calc()
        repeatedly on all input/output ground-truth training data pairs generated by the method 
        gen_gt_dataset().  The call to the forward_propagate...() method returns the predicted value
        at the output nodes from the supplied values at the input nodes.  The "train_on_all_data()"
        method calculates the error associated with the predicted value.  The call to
        forward_propagate...() also returns the partial derivatives estimated by using the finite
        difference method in the computational graph.  Using the partial derivatives, the 
        "train_on_all_data()" backpropagates the loss to the interior nodes in the computational graph
        and updates the values for the learnable parameters.
        '''
        self.vals_for_learnable_params = {var: random.uniform(0,1) for var in self.learnable_params}
        print("\n\nInitial values for all learnable parameters: %s" % str(self.vals_for_learnable_params))
        for sample_index in range(self.dataset_size):
            if (sample_index > 0) and (sample_index % self.display_loss_how_often == 0):
                print("\n\n\n\n============  [Forward Propagation] Training with sample indexed: %d ===============" % sample_index)
            input_vals_for_ind_vars = {var: self.dataset_input_samples[sample_index][var] for var in self.independent_vars}
            if (sample_index > 0) and (sample_index % self.display_loss_how_often == 0):
                print("\ninput values for independent variables: ", input_vals_for_ind_vars)
            predicted_output_vals, partial_var_to_param, partial_var_to_var = \
         self.forward_propagate_one_input_sample_with_partial_deriv_calc(sample_index, input_vals_for_ind_vars)
            error = [self.true_output_vals[sample_index][var] - predicted_output_vals[var] for var in self.output_vars]
            loss = np.linalg.norm(error)
            if self.debug:
                print("\nloss for training sample indexed %d: %s" % (sample_index, str(loss)))
            self.LOSS.append(loss)
            if (sample_index > 0) and (sample_index % self.display_loss_how_often == 0):
                print("\npredicted value at the output nodes: ", predicted_output_vals)
                print("\nloss for training sample indexed %d: %s" % (sample_index, str(loss)))
                print("\nestimated partial derivatives of vars wrt learnable parameters:")
                for k,v in partial_var_to_param.items():
                    print("\nk=%s     v=%s" % (k, str(v)))
                print("\nestimated partial derivatives of vars wrt other vars:")
                for k,v in partial_var_to_var.items():
                    print("\nk=%s     v=%s" % (k, str(v)))
            paths = {param : [] for param in self.learnable_params}
            for var1 in partial_var_to_param:
                for var2 in partial_var_to_param[var1]:
                    for param in self.learnable_params:
                        if partial_var_to_param[var1][var2][param] is not None:
                            paths[param] += [var1,var2,param]
            for param in paths:
                node = paths[param][0]
                if node in self.output_vars: 
                    continue
                for var_out in self.output_vars:        
                    if node in self.depends_on[var_out]:
                        paths[param].insert(0,var_out) 
                    else:
                        for node2 in self.depends_on[var_out]:
                            if node in self.depends_on[node2]:
                                paths[param].insert(0,node2)
                                paths[param].insert(0,var_out)
            for param in self.learnable_params:
                product_of_partials = 1.0
                for i in range(len(paths[param]) - 2):
                    var1 = paths[param][i]
                    var2 = paths[param][i+1]
                    product_of_partials *= partial_var_to_var[var1][var2]
                if self.debug:
                    print("\n\nfor param=%s, product of partials: %s" % str(product_of_partials))
                product_of_partials *=  partial_var_to_param[var1][var2][param]
                self.vals_for_learnable_params[param] -=  self.learning_rate * product_of_partials
            if (sample_index > 0) and (sample_index % self.display_loss_how_often == 0):
                    print("\n\n\n[sample index: %d]: input val: %s    vals for learnable parameters: %s" % (sample_index, str(input_vals_for_ind_vars), str(self.vals_for_learnable_params)))


    def forward_propagate_one_input_sample_with_partial_deriv_calc(self, sample_index, input_vals_for_ind_vars):
        '''
        If you want to look at how the information flows in the DAG when you don't have to worry about
        estimating the partial derivatives, see the method gen_gt_dataset().  As you will notice in the
        implementation code for that method, there is nothing much to pushing the input values through
        the nodes and the arcs of a computational graph if we are not concerned about estimating the
        partial derivatives.

        On the other hand, if you want to see how one might also estimate the partial derivatives as
        during the forward flow of information in a computational graph, the forward_propagate...()
        presented here is the method to examine.  We first split the expression that the node 
        variable depends on into its constituent parts on the basis of '+' and '-' operators and
        subsequently, for each part, we estimate the partial of the node variable with respect
        to the variables and the learnable parameters in that part.
        '''
        predicted_output_vals = {var : None for var in self.output_vars}
        vals_for_dependent_vars = {var: None for var in self.all_vars if var not in self.independent_vars}
        partials_var_to_param = {var : {var : {ele: None for ele in self.learnable_params} for var in self.all_vars} for var in self.all_vars}
        partials_var_to_var =  {var : {var : None for var in self.all_vars} for var in self.all_vars}       
        while any(v is None for v in [vals_for_dependent_vars[x] for x in vals_for_dependent_vars]):
            for var1 in self.all_vars:
                if var1 in self.dependent_vars and vals_for_dependent_vars[var1] is None: continue
                for var2 in self.leads_to[var1]:
                    if any([vals_for_dependent_vars[var] is None for var in self.depends_on[var2] if var not in self.independent_vars]): continue
                    exp = self.expressions_dict[var2]
                    learnable_params_in_exp = [ele for ele in self.learnable_params if ele in exp]
                    ##  in order to calculate the partials of the node (each node stands for a variable)
                    ##  values with respect to the learnable params, and, then, with respect to the 
                    ##  source vars, we must break the exp at '+' and '-' operators:
                    parts =  re.split(r'\+|-', exp)
                    if self.debug:
                        print("\n\n\n\n  ====for var2=%s =================   for exp=%s     parts: %s" % (var2, str(exp), str(parts)))
                    vals_for_parts = []
                    for part in parts:
                        splits_at_arith = re.split(r'\*|/', part)
                        if len(splits_at_arith) > 1:
                            operand1 = splits_at_arith[0]
                            operand2 = splits_at_arith[1]
                            if '^' in operand1:
                                operand1 = operand1[:operand1.find('^')]
                            if '^' in operand2:
                                operand2 = operand2[:operand2.find('^')]
                            if operand1.startswith('x'):
                                var_in_part = operand1
                                param_in_part = operand2
                            elif operand2.startswith('x'):
                                var_in_part = operand2
                                param_in_part = operand1
                            else:
                                sys.exit("you are not following the convention -- aborting")
                        else:
                            if '^' in part:
                                ele_in_part = part[:part.find('^')]
                                if ele_in_part.startswith('x'):
                                    var_in_part = ele_in_part
                                    param_in_part = ""
                                else:
                                    param_in_part = ele_in_part
                                    var_in_part = ""
                            else:
                                if part.startswith('x'):
                                    var_in_part = part
                                    param_in_part = ""
                                else:
                                    param_in_part = part
                                    var_in_part = ""
                        if self.debug:
                            print("\n\n\nvar_in_part: %s    para_in_part=%s" % (var_in_part, param_in_part))
                        part_for_partial_var2var = copy.deepcopy(part)
                        part_for_partial_var2param = copy.deepcopy(part)
                        if self.debug:
                            print("\n\nSTEP1a: part: %s  of   exp: %s" % (part, exp))
                            print("STEP1b: part_for_partial_var2var: %s  of   exp: %s" % (part_for_partial_var2var, exp))
                            print("STEP1c: part_for_partial_var2param: %s  of   exp: %s" % (part_for_partial_var2param, exp))
                        if var_in_part in self.independent_vars:
                            part = part.replace(var_in_part, str(input_vals_for_ind_vars[var_in_part]))
                            part_for_partial_var2var  = part_for_partial_var2var.replace(var_in_part, 
                                                             str(input_vals_for_ind_vars[var_in_part] + self.grad_delta))
                            part_for_partial_var2param = part_for_partial_var2param.replace(var_in_part, 
                                                                               str(input_vals_for_ind_vars[var_in_part]))
                            if self.debug:
                                print("\n\nSTEP2a: part: %s   of   exp=%s" % (part, exp))
                                print("STEP2b: part_for_partial_var2var: %s   of   exp=%s" % (part_for_partial_var2var, exp))
                                print("STEP2c: part_for_partial_var2param: %s   of   exp=%s" % 
                                                                                    (part_for_partial_var2param, exp))
                        if var_in_part in self.dependent_vars:
                            if vals_for_dependent_vars[var_in_part] is not None:
                                part = part.replace(var_in_part, str(vals_for_dependent_vars[var_in_part]))
                                part_for_partial_var2var  = part_for_partial_var2var.replace(var_in_part, 
                                                             str(vals_for_dependent_vars[var_in_part] + self.grad_delta))
                                part_for_partial_var2param = part_for_partial_var2param.replace(var_in_part, 
                                                                               str(vals_for_dependent_vars[var_in_part]))
                            if self.debug:
                                print("\n\nSTEP3a: part=%s   of   exp: %s" % (part, exp))
                                print("STEP3b: part_for_partial_var2var=%s   of   exp: %s" % (part_for_partial_var2var, exp))
                                print("STEP3c: part_for_partial_var2param: %s   of   exp=%s" % (part_for_partial_var2param, exp))
                        ##  now we do the same thing wrt the learnable parameters:
                        if param_in_part != "" and param_in_part in self.learnable_params:
                            if self.vals_for_learnable_params[param_in_part] is not None:
                                part = part.replace(param_in_part, str(self.vals_for_learnable_params[param_in_part]))
                                part_for_partial_var2var  = part_for_partial_var2var.replace(param_in_part, 
                                                                     str(self.vals_for_learnable_params[param_in_part]))
                                part_for_partial_var2param  = part_for_partial_var2param.replace(param_in_part, 
                                                   str(self.vals_for_learnable_params[param_in_part] + self.grad_delta))
                                if self.debug:
                                    print("\n\nSTEP4a: part: %s  of  exp: %s" % (part, exp))
                                    print("STEP4b: part_for_partial_var2var=%s   of   exp: %s" % 
                                                                                       (part_for_partial_var2var, exp))
                                    print("STEP4c: part_for_partial_var2param=%s   of   exp: %s" % 
                                                                                      (part_for_partial_var2param, exp))
                        ###  Now evaluate the part for each of three cases:
                        evaled_part = eval( part.replace('^', '**') )
                        vals_for_parts.append(evaled_part)
                        evaled_partial_var2var = eval( part_for_partial_var2var.replace('^', '**') )
                        if param_in_part != "":
                            evaled_partial_var2param = eval( part_for_partial_var2param.replace('^', '**') )
                        partials_var_to_var[var2][var_in_part] = (evaled_partial_var2var - evaled_part) / self.grad_delta
                        if param_in_part != "":
                            partials_var_to_param[var2][var_in_part][param_in_part] = \
                                                               (evaled_partial_var2param - evaled_part) / self.grad_delta
                    vals_for_dependent_vars[var2] = sum(vals_for_parts)
        predicted_output_vals = {var : vals_for_dependent_vars[var] for var in self.output_vars}
        return predicted_output_vals, partials_var_to_param, partials_var_to_var


    def forward_propagate_with_partial_deriv_calc(self, input_vals_for_ind_vars, class_label):
        input_vars = self.independent_vars
        vals_for_input_vars =  dict(zip(self.independent_vars, list(input_vals_for_ind_vars)))
        print("\n\nvals_for_input_vars: ", vals_for_input_vars)      

        partials_var_to_param = {var : {ele: None for ele in self.learnable_params} for var in self.all_vars} 

        partials_var_to_var =  {var : {var : None for var in self.all_vars} for var in self.all_vars}       
        input_vars = self.independent_vars

        exp_obj = self.exp_objects[0]
        output_val = self.eval_expression(exp_obj.body , vals_for_input_vars, 
                                                               self.vals_for_learnable_params)
        ## apply ReLU activation:
        output_val = output_val if output_val > 0 else 0
        input_vars = exp_obj.right_vars
        input_params = exp_obj.right_params
        output_var = exp_obj.dependent_var
        print("\n\noutput_val: ", output_val)
        print("\n\npartials_output_var_to_vars: ", exp_obj.partials_output_var_to_vars)
        partials_output_var_to_vars = exp_obj.partials_output_var_to_vars
        print("\n\npartials_output_var_to_params: ", exp_obj.partials_output_var_to_params)
        partials_output_var_to_params = exp_obj.partials_output_var_to_params
        target = class_label
        loss = (target - output_val)**2
        print("\n\nloss: ", loss)
        partials_output_var_to_vars_with_vals =  exp_obj.partials_output_var_to_vars_with_vals
        partials_output_var_to_params_with_vals = exp_obj.partials_output_var_to_params_with_vals
        for var1 in partials_output_var_to_vars:
            for var2 in partials_output_var_to_vars[var1]:
                partials_output_var_to_vars_with_vals[var1][var2] = self.eval_expression(
                                                                partials_output_var_to_vars[var1][var2],
                                                                vals_for_input_vars,
                                                                self.vals_for_learnable_params)
        for var in partials_output_var_to_params:
            for param in partials_output_var_to_params[var]:
                partials_output_var_to_params_with_vals[var][param] = self.eval_expression(
                                                                partials_output_var_to_params[var][param],
                                                                vals_for_input_vars,
                                                                self.vals_for_learnable_params)

        print("\n\npartials_output_var_to_vars_with_vals: ", partials_output_var_to_vars_with_vals)
        print("\n\npartials_output_var_to_params_with_vals: ", partials_output_var_to_params_with_vals)

        return {'Loss' :  loss, 
                'partials_output_var_to_vars_with_vals': partials_output_var_to_vars_with_vals, 
                'partials_output_var_to_params_with_vals': partials_output_var_to_params_with_vals}
    ######################################################################################################


    ## New in Version 1.0.6:
    ######################################################################################################
    #######################  Start Definition of Inner Class AutogradCustomization  ######################
    class AutogradCustomization(torch.nn.Module):             
        """
        This class illustrates how you can add additional functionality of Autograd by 
        following the instructions posted at
                   https://pytorch.org/docs/stable/notes/extending.html
        """

        def __init__(self, cgp, num_samples_per_class):
            super(ComputationalGraphPrimer.AutogradCustomization, self).__init__()
            self.cgp = cgp
            self.num_samples_per_class = num_samples_per_class


        class DoSillyWithTensor(torch.autograd.Function):                  
            """        
            Extending Autograd requires that you define a new verb class, as I have with
            the class DoSillyWithTensor shown here, with definitions for two static
            methods, "forward()" and "backward()".  An instance constructed from this
            class is callable.  So when, in the "forward()" of the network, you pass a
            training sample through an instance of DoSillyWithTensor, it is subject to
            the code shown below in the "forward()"  of this class.
            """
            @staticmethod
            def forward(ctx, input):
                """
                The parameter 'input' is set to the training sample that is being 
                processed by an instance of DoSillyWithTensor in the "forward()" of a
                network.  We first make a deep copy of this tensor (which should be a 
                32-bit float) and then we subject the copy to a conversion to a one-byte 
                integer, which should cause a significant loss of information. We 
                calculate the difference between the original 32-bit float and the 8-bit 
                version and store it away in the context variable "ctx".
                """
                input_orig = input.clone().double()
                input = input.to(torch.uint8).double()
                diff = input_orig.sub(input)
                ctx.save_for_backward(diff)
                return input

            @staticmethod
            def backward(ctx, grad_output):
                """
                Whatever was stored in the context variable "ctx" during the forward pass
                can be retrieved in the backward pass as shown below.
                """
                diff, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input = grad_input + diff
                return grad_input
        
        def gen_training_data(self):        
            mean1,mean2   = [3.0,3.0], [5.0,5.0]
            covar1,covar2 = [[1.0,0.0], [0.0,1.0]], [[1.0,0.0], [0.0,1.0]]
            data1 = [(list(x),1) for x in np.random.multivariate_normal(mean1, covar1, self.num_samples_per_class)]
            data2 = [(list(x),2) for x in np.random.multivariate_normal(mean2, covar2, self.num_samples_per_class)]
            training_data = data1 + data2
            random.shuffle( training_data )
            self.training_data = training_data 

        def train_with_straight_autograd(self):
            dtype = torch.float
            D_in,H,D_out = 2,10,2
#           w1 = torch.randn(D_in, H, device="cpu", dtype=dtype, requires_grad=True)
#           w2 = torch.randn(H, D_out, device="cpu", dtype=dtype, requires_grad=True)
            w1 = torch.randn(D_in, H, device="cpu", dtype=dtype)
            w2 = torch.randn(H, D_out, device="cpu", dtype=dtype)
            w1 = w1.to(self.cgp.device)
            w2 = w2.to(self.cgp.device)
            w1.requires_grad_()
            w2.requires_grad_()
            Loss = []
            for epoch in range(self.cgp.epochs):
                for i,data in enumerate(self.training_data):
                    input, label = data
                    x,y = torch.as_tensor(np.array(input)), torch.as_tensor(np.array(label))
                    x,y = x.float(), y.float()
                    if self.cgp.device:
                        x,y = x.to(self.cgp.device), y.to(self.cgp.device)
                    y_pred = x.view(1,-1).mm(w1).clamp(min=0).mm(w2)
                    loss = (y_pred - y).pow(2).sum()
                    if i % 200 == 199:
                        Loss.append(loss.item())
                        print("epoch=%d i=%d" % (epoch,i), loss.item())
#                   w1.retain_grad()
#                   w2.retain_grad()
                    loss.backward()       
                    with torch.no_grad():
                        w1 -= self.cgp.learning_rate * w1.grad
                        w2 -= self.cgp.learning_rate * w2.grad
                        w1.grad.zero_()
                        w2.grad.zero_()
            print("\n\n\nLoss: %s" % str(Loss))
            import matplotlib.pyplot as plt
            plt.figure("Loss vs training (straight autograd)")
            plt.plot(Loss)
            plt.show()

        def train_with_extended_autograd(self):
            dtype = torch.float
            D_in,H,D_out = 2,10,2
#           w1 = torch.randn(D_in, H, device="cpu", dtype=dtype, requires_grad=True)
#           w2 = torch.randn(H, D_out, device="cpu", dtype=dtype, requires_grad=True)
            w1 = torch.randn(D_in, H, device="cpu", dtype=dtype)
            w2 = torch.randn(H, D_out, device="cpu", dtype=dtype)
            w1 = w1.to(self.cgp.device)
            w2 = w2.to(self.cgp.device)
            w1.requires_grad_()
            w2.requires_grad_()
            Loss = []
            for epoch in range(self.cgp.epochs):
                for i,data in enumerate(self.training_data):
                    ## Constructing an instance of DoSillyWithTensor. It is callable.
                    do_silly = ComputationalGraphPrimer.AutogradCustomization.DoSillyWithTensor.apply      
                    input, label = data
                    x,y = torch.as_tensor(np.array(input)), torch.as_tensor(np.array(label))
                    ## Now process the training instance with the "do_silly" instance:
                    x = do_silly(x)                                 
                    x,y = x.float(), y.float()
                    x,y = x.to(self.cgp.device), y.to(self.cgp.device)
                    y_pred = x.view(1,-1).mm(w1).clamp(min=0).mm(w2)
                    loss = (y_pred - y).pow(2).sum()
                    if i % 200 == 199:
                        Loss.append(loss.item())
                        print("epoch=%d i=%d" % (epoch,i), loss.item())
#                   w1.retain_grad()
#                   w2.retain_grad()
                    loss.backward()       
                    with torch.no_grad():
                        w1 -= self.cgp.learning_rate * w1.grad
                        w2 -= self.cgp.learning_rate * w2.grad
                        w1.grad.zero_()
                        w2.grad.zero_()
            print("\n\n\nLoss: %s" % str(Loss))
            import matplotlib.pyplot as plt
            plt.figure("loss vs training (extended autograd)")
            plt.plot(Loss)
            plt.show()
    ######################################################################################################


    ######################################################################################################
    ######################################  Utility Functions ############################################
    def calculate_loss(self, predicted_val, true_val):
        error = true_val - predicted_val
        loss = np.linalg.norm(error)
        return loss

    def plot_loss(self):
        plt.figure()
        plt.plot(self.LOSS)
        plt.show()

    def eval_expression(self, exp, vals_for_vars, vals_for_learnable_params, ind_vars=None):
        self.debug1 = False
        if self.debug1:
            print("\n\nSTEP1: [original expression] exp: %s" % exp)
        if ind_vars is not None:
            for var in ind_vars:
                exp = exp.replace(var, str(vals_for_vars[var]))
        else:
            for var in self.independent_vars:
                exp = exp.replace(var, str(vals_for_vars[var]))
        if self.debug1:
            print("\n\nSTEP2: [replaced ars by their vals] exp: %s" % exp)
        for ele in self.learnable_params:
            exp = exp.replace(ele, str(vals_for_learnable_params[ele]))
        if self.debug1:                     
            print("\n\nSTEP4: [replaced learnable params by their vals] exp: %s" % exp)
        return eval( exp.replace('^', '**') )


    def gen_gt_dataset(self, vals_for_learnable_params={}):
        '''
        This method illustrates that it is trivial to forward-propagate the information through
        the computational graph if you are not concerned about estimating the partial derivatives
        at the same time.  This method is used to generate 'dataset_size' number of input/output
        values for the computational graph for given values for the learnable parameters.
        '''
        N = self.dataset_size
        for i in range(N):
            if self.debug:
                print("\n\n\n================== Gen GT: iteration %d ============================\n" % i)
#            vals_for_ind_vars = {var: random.uniform(0,1) for var in self.independent_vars}
            vals_for_ind_vars = {var: random.uniform(-1,1) for var in self.independent_vars}
            self.dataset_input_samples[i] = vals_for_ind_vars    
            vals_for_dependent_vars = {var: None for var in self.all_vars if var not in self.independent_vars}
            while True:
                if not any(v is None for v in [vals_for_dependent_vars[x] for x in vals_for_dependent_vars]):
                    break
                for var1 in self.all_vars:
                    for var2 in self.leads_to[var1]:
                        if vals_for_dependent_vars[var2] is not None: continue
                        predecessor_vars = self.depends_on[var2]
                        predecessor_vars_without_inds = [x for x in predecessor_vars if x not in self.independent_vars]
                        if any(vals_for_dependent_vars[vart] is None for vart in predecessor_vars_without_inds): continue
                        exp = self.expressions_dict[var2]
                        if self.debug:
                            print("\n\nSTEP1: [original expression] exp: %s" % exp)
                        for var in self.independent_vars:
                            exp = exp.replace(var, str(vals_for_ind_vars[var]))
                        if self.debug:
                            print("\n\nSTEP2: [replaced independent vars by their vals] exp: %s" % exp)
                        for var in self.dependent_vars:
                            if vals_for_dependent_vars[var] is not None:
                                exp = exp.replace(var, str(vals_for_dependent_vars[var]))
                        if self.debug:
                            print("\n\nSTEP3: [replaced dependent vars by their vals] exp: %s" % exp)
                        for ele in self.learnable_params:
                            exp = exp.replace(ele, str(vals_for_learnable_params[ele]))
                        if self.debug:                     
                            print("\n\nSTEP4: [replaced learnable params by their vals] exp: %s" % exp)
                        vals_for_dependent_vars[var2] = eval( exp.replace('^', '**') )
            self.true_output_vals[i] = {ovar : vals_for_dependent_vars[ovar] for ovar in self.output_vars}
        if self.debug:
            print("\n\n\ninput samples: %s" % str(self.dataset_input_samples))
            print("\n\n\noutput vals: %s" % str(self.true_output_vals))


    def gen_training_data(self):
        num_input_vars = len(self.independent_vars)
        training_data_class_0 = []
        training_data_class_1 = []
        for i in range(self.dataset_size //2):
            # Standard normal means N(0,1), meaning zero mean and unit variance
            # Such values are significant in the interval [-3.0,+3.0]
            for_class_0 = np.random.standard_normal( num_input_vars )
            for_class_1 = np.random.standard_normal( num_input_vars )
            # translate class_1 data so that the mean is shifted to +4.0 and also
            # change the variance:
            for_class_0 = for_class_0 + 2.0
            for_class_1 = for_class_1 * 2 + 4.0
            training_data_class_0.append( for_class_0 )
            training_data_class_1.append( for_class_1 )
        self.training_data = {0 : training_data_class_0, 1 : training_data_class_1}


    def gen_gt_dataset_with_activations(self, vals_for_learnable_params={}):
        '''
        This method illustrates that it is trivial to forward-propagate the information through
        the computational graph if you are not concerned about estimating the partial derivatives
        at the same time.  This method is used to generate 'dataset_size' number of input/output
        values for the computational graph for given values for the learnable parameters.
        '''
        N = self.dataset_size
        for i in range(N):
            if self.debug:
                print("\n\n\n================== Gen GT: iteration %d ============================\n" % i)
#            vals_for_ind_vars = {var: random.uniform(0,1) for var in self.independent_vars}
            vals_for_ind_vars = {var: random.uniform(-1,1) for var in self.independent_vars}
            self.dataset_input_samples[i] = vals_for_ind_vars    
            vals_for_dependent_vars = {var: None for var in self.all_vars if var not in self.independent_vars}
            while True:
                if not any(v is None for v in [vals_for_dependent_vars[x] for x in vals_for_dependent_vars]):
                    break
                for var1 in self.all_vars:
                    for var2 in self.leads_to[var1]:
                        if vals_for_dependent_vars[var2] is not None: continue
                        predecessor_vars = self.depends_on[var2]
                        predecessor_vars_without_inds = [x for x in predecessor_vars if x not in self.independent_vars]
                        if any(vals_for_dependent_vars[vart] is None for vart in predecessor_vars_without_inds): continue
                        exp = self.expressions_dict[var2]
                        if self.debug:
                            print("\n\nSTEP1: [original expression] exp: %s" % exp)
                        for var in self.independent_vars:
                            exp = exp.replace(var, str(vals_for_ind_vars[var]))
                        if self.debug:
                            print("\n\nSTEP2: [replaced independent vars by their vals] exp: %s" % exp)
                        for var in self.dependent_vars:
                            if vals_for_dependent_vars[var] is not None:
                                exp = exp.replace(var, str(vals_for_dependent_vars[var]))
                        if self.debug:
                            print("\n\nSTEP3: [replaced dependent vars by their vals] exp: %s" % exp)
                        for ele in self.learnable_params:
                            exp = exp.replace(ele, str(vals_for_learnable_params[ele]))
                        if self.debug:                     
                            print("\n\nSTEP4: [replaced learnable params by their vals] exp: %s" % exp)
                        vals_for_dependent_vars[var2] = eval( exp.replace('^', '**') )
                        ## ReLU activation:
                        vals_for_dependent_vars[var2] = 0 if vals_for_dependent_vars[var2] < 0 else vals_for_dependent_vars[var2]

            self.true_output_vals[i] = {ovar : vals_for_dependent_vars[ovar] for ovar in self.output_vars}
        if self.debug:
            print("\n\n\ninput samples: %s" % str(self.dataset_input_samples))
            print("\n\n\noutput vals: %s" % str(self.true_output_vals))

#_________________________  End of ComputationalGraphPrimer Class Definition ___________________________


#______________________________    Test code follows    _________________________________

if __name__ == '__main__': 
    pass
