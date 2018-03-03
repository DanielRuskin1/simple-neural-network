# Simple Neural Network

A program that trains a simple feedforward neural network to fit training data, evaluates the network against test data, and saves the model to a file.

For more details about the CLI, run `./simple_neural_network help`.

# Neural Network Properties

This software will generate a fully-connected, feedforward neural network and will train it via backpropagation and stochastic gradient descent.  It can generate a network with an arbitrary number of layers/nodes; the "layers" parameter is simply a list of the number of nodes for each layer.

The network uses a mean-squared-error cost function for training.  It uses the sigmoid activation function to connect nodes.  Both of these properties are trivially updateable; one can simply create new cost/activation classes that implement `eval` and `evalPrime`, and initialize `NeuralNetwork`/`NeuralNetworkTrainer` with those new classes as template variables.

# UML

A UML diagram for the program is included in the `uml` folder.

# Example Command

```
./simple_neural_network --layers "784 30 30 10" --training_set_features_file ../mnist_data/training_features.csv --training_set_labels_file ../mnist_data/training_labels.csv --test_set_features_file ../mnist_data/test_features.csv --test_set_labels_file ../mnist_data/test_labels.csv --learning_rate 3.0 --samples_per_epoch -10 --num_epochs 3 --output_prefix "./results/"
```

This will train the neural network on the MNIST dataset.  It ends up achieving about 94.4% accuracy after just 3 epochs.

# Dependencies

This program requires Boost (>= 1.60.0).

# Scripts

Several Ruby scripts are included in the `scripts` directory.  These may be helpful for transforming your data.

# References

The following references were used while preparing this program:

```
Goodfellow, Ian, et al. Deep learning. MIT Press, 2016.
Nielson, Michael A. Neural Networks and Deep Learning. Determination Press, 2015.
```
