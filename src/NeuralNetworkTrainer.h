/*
 * NeuralNetworkTrainer.h
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#ifndef SRC_NEURALNETWORKTRAINER_H_
#define SRC_NEURALNETWORKTRAINER_H_

#include <vector>
#include <armadillo>
#include "NeuralNetwork.h"

template <class Activation, class Cost>
class NeuralNetworkTrainer {
public:
	typedef NeuralNetwork<Activation, Cost> NeuralNetworkLoc;
	NeuralNetworkTrainer(std::shared_ptr<NeuralNetworkLoc> new_network, int new_mini_batch_size, int new_num_epochs);

	void trainNetwork() const;
private:
	std::shared_ptr<NeuralNetworkLoc> network;
	int mini_batch_size;
	int num_epochs;

	void calcGradients(const VecOfColVecs& activations,
					   const VecOfColVecs& weighted_inputs,
					   std::unique_ptr<VecOfMats>& out_weight_gradients,
					   std::unique_ptr<VecOfColVecs>& out_bias_gradients) const;
};

#endif /* SRC_NEURALNETWORKTRAINER_H_ */
