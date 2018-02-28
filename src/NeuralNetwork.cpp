/*
 * NeuralNetwork.cpp
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#include "NeuralNetwork.h"

template <class Activation, class Cost>
NeuralNetwork<Activation, Cost>::NeuralNetwork(const VecOfInts& config) {
	num_layers = config.size();
	input_layer_size = config[0];
	weights.resize(num_layers - 1);
	biases.resize(num_layers - 1);

	// Weights/biases for all layers other than the first one
	// Weird indexing because 0th layer has no weights/biases:
	// 		Weight[L](J, K) is connecting node J in layer L + 1 to node K in layer L;
	// 		Bias[L](J) is bias for node J in layer L + 1.
	// Abstracted away later via getter methods
	for(int l = 0; l < config.size() - 1; l++) {
		// Fill to [0.01, 1.01] and divide by 10 => [0.001, 0.101]
		weights[l].resize(config[l + 1], config[l]);
		weights[l].fill(arma::fill::randn);
		weights[l] += 0.01;
		weights[l] /= 10;

		biases[l].resize(config[l + 1]);
		biases[l].fill(arma::fill::randn);
		biases[l] += 0.01;
		biases[l] /= 10;
	}
}

// Returns weights connecting layer L to layer L - 1
// Indexing starts with first non-input layer (so you cannot call this for the input layer).
template <class Activation, class Cost>
const arma::mat& NeuralNetwork<Activation, Cost>::getWeights(int layer) const {
	return weights[layer];
}

// Returns biases for layer L.
// Undefined behavior if called for layer 0.
// Indexing starts with first non-input layer (so you cannot call this for the input layer).
template <class Activation, class Cost>
const arma::colvec& NeuralNetwork<Activation, Cost>::getBiases(int layer) const {
	return biases[layer];
}

// Returned weighted_inputs starts with first hidden layer (0th element => initial weighted input).
// Return activations starts with input layer (0th element => input).
template <class Activation, class Cost>
void NeuralNetwork<Activation, Cost>::feedForward(const arma::colvec& input, std::unique_ptr<VecOfColVecs>& out_weighted_inputs, std::unique_ptr<VecOfColVecs>& out_activations) const {
	arma::colvec last_activation = input;
	if(last_activation.size() != input_layer_size) { throw std::runtime_error("Invalid input size!"); }

	out_weighted_inputs.reset(new VecOfColVecs);
	out_activations.reset(new VecOfColVecs);

	// First activation is technically the input
	out_activations->push_back(input);

	for(int l = 1; l < num_layers; l++) {
		arma::colvec weighted_input = (getWeights(l - 1) * last_activation) + getBiases(l - 1);
		std::unique_ptr<arma::colvec> activated = Activation::eval(weighted_input);
		last_activation = *activated;

		out_weighted_inputs->push_back(weighted_input);
		out_activations->push_back(last_activation);
	}
}

template <class Activation, class Cost>
std::unique_ptr<arma::colvec> NeuralNetwork<Activation, Cost>::predict(const arma::colvec& input) const {
	std::unique_ptr<arma::colvec> out;

	std::unique_ptr<VecOfColVecs> outWI;
	std::unique_ptr<VecOfColVecs> outAct;

	feedForward(input, outWI, outAct);

	out.reset(new arma::colvec(outAct->back()));
	return std::move(out);
}

// Updates weights connecting layer L to layer L - 1
// Updates biases for layer L.
// Indexing starts with first non-input layer (so you cannot call this for the input layer).
template <class Activation, class Cost>
void NeuralNetwork<Activation, Cost>::setLayerProperties(int layer, const arma::mat& new_weights, const arma::colvec& new_biases) {
	weights[layer] = new_weights;
	biases[layer] = new_biases;
}
