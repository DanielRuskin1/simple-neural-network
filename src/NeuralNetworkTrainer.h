/*
 * NeuralNetworkTrainer.h
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#ifndef SRC_NEURALNETWORKTRAINER_H_
#define SRC_NEURALNETWORKTRAINER_H_

#include <numeric>
#include <vector>
#include <armadillo>
#include <boost/log/trivial.hpp>
#include "NeuralNetwork.h"

template <class Activation, class Cost>
class NeuralNetworkTrainer {
public:
	typedef NeuralNetwork<Activation, Cost> NeuralNetworkLoc;
	NeuralNetworkTrainer(std::shared_ptr<NeuralNetworkLoc> new_network,
						 std::shared_ptr<const arma::mat> new_training_features,
						 std::shared_ptr<const arma::mat> new_training_labels,
						 std::shared_ptr<const arma::mat> new_test_features,
						 std::shared_ptr<const arma::mat> new_test_labels,
						 double new_learning_rate,
						 int new_mini_batch_size,
						 int new_num_epochs);

	void trainNetwork() const;
private:
	std::shared_ptr<NeuralNetworkLoc> network;
	std::shared_ptr<const arma::mat> training_features;
	std::shared_ptr<const arma::mat> training_labels;
	std::shared_ptr<const arma::mat> test_features;
	std::shared_ptr<const arma::mat> test_labels;
	double learning_rate;
	int mini_batch_size;
	int num_epochs;

	void calcGradients(const VecOfColVecs& weighted_inputs,
					   const VecOfColVecs& activations,
					   const arma::colvec& correct_val,
					   std::unique_ptr<VecOfMats>& out_weight_gradients,
					   std::unique_ptr<VecOfColVecs>& out_bias_gradients) const;
};

#endif /* SRC_NEURALNETWORKTRAINER_H_ */
