/*
 * NeuralNetwork.cpp
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#include "NeuralNetwork.h"

template <class Activation, class Cost> NeuralNetwork<Activation, Cost>::NeuralNetwork(const VecOfInts& config) {
	num_layers = config.size();
	input_layer_size = config[0];
	weights.resize(num_layers);
	biases.resize(num_layers);

	// Weights/biases for first layer
	weights[0].resize(config[0], 1);
	weights[0].ones();
	biases[0].resize(config[0], 1);
	biases[0].zeros();

	// Weights/biases for all other layers
	// Weight[L](J, K) is connecting node J in layer L to node K in layer L - 1;
	// Bias[L](J) is bias for node J in layer L.
	for(int l = 1; l < config.size(); l++) {
		// Fill to [0.01, 1.01] and divide by 10 => [0.001, 0.101]
		weights[l].resize(config[l], config[l - 1]);
		weights[l].fill(arma::fill::randn);
		weights[l] += 0.01;
		weights[l] /= 10;

		biases[l].resize(config[l]);
		biases[l].fill(arma::fill::randn);
		biases[l] += 0.01;
		biases[l] /= 10;
	}
}

template <class Activation, class Cost> void NeuralNetwork<Activation, Cost>::feedForward(const arma::colvec& input, std::unique_ptr<VecOfColVecs>& out_weighted_inputs, std::unique_ptr<VecOfColVecs>& out_activations) const {
	arma::colvec last_activation = input;
	if(last_activation.size() != input_layer_size) { throw std::runtime_error("Invalid input size!"); }

	out_weighted_inputs.reset(new VecOfColVecs);
	out_activations.reset(new VecOfColVecs);

	for(int l = 0; l < num_layers; l++) {
		arma::colvec weighted_input = (last_activation * weights[l]) + biases[l];
		arma::colvec activated = Activation::eval(weighted_input);

		out_weighted_inputs->push_back(weighted_input);
		out_activations->push_back(activated);

		last_activation = activated;
	}
}

template <class Activation, class Cost> std::unique_ptr<arma::colvec> NeuralNetwork<Activation, Cost>::predict(const arma::colvec& input) const {
	std::unique_ptr<arma::colvec> out;

	std::unique_ptr<VecOfColVecs> outWI;
	std::unique_ptr<VecOfColVecs> outAct;

	feedForward(input, outWI, outAct);

	out.reset(new arma::colvec(outAct->back()));
	return std::move(out);
}
