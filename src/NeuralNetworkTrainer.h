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

namespace SimpleNeuralNetwork {

class NeuralNetworkTrainer {
public:
	NeuralNetworkTrainer(std::shared_ptr<NeuralNetwork> new_network, int new_mini_batch_size, int new_num_epochs);

	void trainNetwork() const;
private:
	std::shared_ptr<NeuralNetwork> network;
	int mini_batch_size;
	int num_epochs;
};

} /* namespace SimpleNeuralNetwork */

#endif /* SRC_NEURALNETWORKTRAINER_H_ */
