/*
 * NeuralNetworkTrainer.cpp
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#include "NeuralNetworkTrainer.h"

template <class Activation, class Cost>
NeuralNetworkTrainer<Activation, Cost>::NeuralNetworkTrainer(std::shared_ptr<NeuralNetworkLoc> new_network, std::shared_ptr<const TrainingExamples> new_examples, double new_learning_rate, int new_mini_batch_size, int new_num_epochs)
	: network(new_network), examples(new_examples), learning_rate(new_learning_rate), mini_batch_size(new_mini_batch_size), num_epochs(new_num_epochs) {

}

template <class Activation, class Cost>
void NeuralNetworkTrainer<Activation, Cost>::trainNetwork() const {
	BOOST_LOG_TRIVIAL(info) << "Training neural network...";

	int num_batches_per_epoch = std::floor(examples->size() / mini_batch_size);
	if(num_batches_per_epoch == 0) { num_batches_per_epoch = 1; }

	for(int epoch = 0; epoch < num_epochs; epoch++) {
		std::string log_msg = "Training epoch " + std::to_string(epoch);
		BOOST_LOG_TRIVIAL(info) << log_msg;

		std::vector<int> examples_remaining_for_epoch(examples->size());
		std::iota(examples_remaining_for_epoch.begin(), examples_remaining_for_epoch.end(), 0);
		std::random_shuffle(examples_remaining_for_epoch.begin(), examples_remaining_for_epoch.end());

		for(int batch = 0; batch < num_batches_per_epoch; batch++) {
			std::string log_msg_deep = log_msg + ", batch " + std::to_string(batch);
			BOOST_LOG_TRIVIAL(info) << log_msg_deep;

			// If on last batch, use all remaining examples in this batch.
			// Otherwise, use normal batch size.
			int num_examples_in_batch;
			if(batch == num_batches_per_epoch - 1) { num_examples_in_batch = examples_remaining_for_epoch.size(); }
			else { num_examples_in_batch = mini_batch_size; }

			// Calculate all gradients for this example + total cost
			VecOfMats weight_grad;
			VecOfColVecs bias_grad;
			double cost = 0;
			for(int ex = 0; ex < num_examples_in_batch; ex++) {
				// Pick example to use
				int exNum = examples_remaining_for_epoch.back();
				examples_remaining_for_epoch.pop_back(); // "use up" this example

				// Feedforward w/ example
				std::unique_ptr<VecOfColVecs> weighted_inputs;
				std::unique_ptr<VecOfColVecs> activations;
				network->feedForward((*examples)[exNum].first, weighted_inputs, activations);

				// Calculate and save gradients
				std::unique_ptr<VecOfMats> weight_grad_loc;
				std::unique_ptr<VecOfColVecs> bias_grad_loc;
				calcGradients(*weighted_inputs, *activations, (*examples)[exNum].second, weight_grad_loc, bias_grad_loc);
				for(unsigned int layer = 0; layer < weight_grad_loc->size(); layer++) {
					if(ex == 0) {
						weight_grad.push_back((*weight_grad_loc)[layer]);
						bias_grad.push_back((*bias_grad_loc)[layer]);
					} else {
						weight_grad[layer] += (*weight_grad_loc)[layer];
						bias_grad[layer] += (*bias_grad_loc)[layer];
					}
				}

				// Calc cost
				cost += Cost::eval(activations->back(), (*examples)[exNum].second);
			}
			BOOST_LOG_TRIVIAL(info) << log_msg_deep << ", calculated cost/cost gradients!";
			BOOST_LOG_TRIVIAL(info) << log_msg_deep << ", cost before this epoch: " << cost;

			// Update weight/bias on each layer
			for(unsigned int layer = 0; layer < weight_grad.size(); layer++) {
				// Calc avg for SGD
				weight_grad[layer] /= num_examples_in_batch;
				bias_grad[layer] /= num_examples_in_batch;

				// Update network
				network->setLayerProperties(
					layer,
					network->getWeights(layer) - (weight_grad[layer] * learning_rate),
					network->getBiases(layer) - (bias_grad[layer] * learning_rate)
				);
			}
		}
	}
}

template <class Activation, class Cost>
void NeuralNetworkTrainer<Activation, Cost>::calcGradients(const VecOfColVecs& weighted_inputs,
		   	   	   	   	   	   	   	   	   	   	   	   	   const VecOfColVecs& activations,
														   const arma::colvec& correct_val,
														   std::unique_ptr<VecOfMats>& out_weight_gradients,
														   std::unique_ptr<VecOfColVecs>& out_bias_gradients) const {
	out_weight_gradients.reset(new VecOfMats);
	out_bias_gradients.reset(new VecOfColVecs);

	std::unique_ptr<arma::colvec> tmp_a;
	std::unique_ptr<arma::colvec> tmp_b;

	arma::colvec error_in_next_layer;

	for(int layer = weighted_inputs.size() - 1; layer >= 0; layer--) {
		arma::colvec error_in_this_layer;
		if(layer == weighted_inputs.size() - 1) {
			// For last layer, error is:
			// dC/da * da/dz = dC/dz = error
			tmp_a = Cost::evalPrime(activations.back(), correct_val);
			tmp_b = Activation::evalPrime(weighted_inputs.back());
			error_in_this_layer = (*tmp_a) % (*tmp_b);
		} else {
			// For prev layers, error is made up of:
			// (1) Weights from next layer, transposed (derivative of next layer weighted-input in terms of this layer activation)
			// (2) Error in next layer (derivative of cost in terms of weighted-input)
			// (3) Derivative of this layer activation in terms of this layer input
			// (1 * 2) hadamard 3 => error in this layer (derivative of cost in terms of weighted input)
			tmp_a = Activation::evalPrime(weighted_inputs[layer]);
			error_in_this_layer = (arma::trans(network->getWeights(layer + 1)) * error_in_next_layer) % *tmp_a;
		}

		// Bias gradient is just error in this layer
		out_bias_gradients->push_back(error_in_this_layer);

		// Weight gradient for weight that connects J node here to K node in last layer is:
		// 1. The error in the Jth node here (dC/dZ), times
		// 2. The activation of the Kth node in the last layer (dZ/dW)
		// => dC/dw
		// +1 b/c activations uses indexing INCLUDING the input layer (whereas the 0th element is the first hidden layer here)
		int num_nodes_this = activations[layer + 1].size();
		int num_nodes_last = activations[layer].size();
		out_weight_gradients->push_back(arma::mat(num_nodes_this, num_nodes_last));
		for(int j = 0; j < num_nodes_this; j++) {
			for(int k = 0; k < num_nodes_last; k++) {
				out_weight_gradients->back()(j, k) = error_in_this_layer(j) * activations[layer](k);
			}
		}

		error_in_next_layer = error_in_this_layer; // For next iteration
	}

	// Last layer will be in 0th index of return vals, so we need to reverse.
	std::reverse(out_bias_gradients->begin(), out_bias_gradients->end());
	std::reverse(out_weight_gradients->begin(), out_weight_gradients->end());
}
