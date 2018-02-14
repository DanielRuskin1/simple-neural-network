/*
 * NeuralNetwork.h
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#ifndef SRC_NEURALNETWORK_H_
#define SRC_NEURALNETWORK_H_

#include <vector>
#include <armadillo>

namespace SimpleNeuralNetwork {

class NeuralNetwork {
public:
	NeuralNetwork(const std::vector<int>& configuration);

	void feedForward(const arma::vec& input, std::unique_ptr<arma::mat>& outActivations, std::unique_ptr<arma::mat>& outWeightedInputs) const;
	std::unique_ptr<arma::vec> predict(const arma::vec& input) const;

	const arma::mat& getWeights(int layer) const;
	const arma::mat& getBiases(int layer) const;

	struct NodeProperties {
		double weight;
		double bias;
	};
	void setNodeProperties(int layer, int node, const NodeProperties& props);
private:
	std::vector<arma::mat> weights;
	std::vector<arma::mat> biases;
};

} /* namespace SimpleNeuralNetwork */

#endif /* SRC_NEURALNETWORK_H_ */
