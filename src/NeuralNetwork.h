/*
 * NeuralNetwork.h
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#ifndef SRC_NEURALNETWORK_H_
#define SRC_NEURALNETWORK_H_

#include <memory>
#include <vector>
#include <armadillo>
#include "Utils.h"

template <class Activation, class Cost>
class NeuralNetwork {
public:
	NeuralNetwork(const VecOfInts& config);

	void feedForward(const arma::colvec& input, std::unique_ptr<VecOfColVecs>& out_weighted_inputs, std::unique_ptr<VecOfColVecs>& out_activations) const;
	std::unique_ptr<arma::colvec> predict(const arma::colvec& input) const;

	const arma::mat& getWeights(int layer) const;
	const arma::colvec& getBiases(int layer) const;

	void setLayerProperties(int layer, const arma::mat& new_weights, const arma::colvec& new_biases);
private:
	unsigned int num_layers;
	unsigned int input_layer_size;

	VecOfMats weights;
	VecOfColVecs biases;
};

#endif /* SRC_NEURALNETWORK_H_ */
