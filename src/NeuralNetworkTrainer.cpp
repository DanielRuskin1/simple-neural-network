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

	for(int epoch = 0; epoch < num_epochs; epoch++) {
		BOOST_LOG_TRIVIAL(info) << "Training epoch " << epoch;

		std::vector<int> examples_remaining_for_epoch(examples->size());
		std::iota(examples_remaining_for_epoch.begin(), examples_remaining_for_epoch.end(), 0);
		std::random_shuffle(examples_remaining_for_epoch.begin(), examples_remaining_for_epoch.end());

		for(int batch = 0; batch < num_batches_per_epoch; batch++) {
			// If on last batch, use all remaining examples in this batch.
			// Otherwise, use normal batch size.
			int num_examples_in_batch;
			if(batch == num_batches_per_epoch - 1) { num_examples_in_batch = examples_remaining_for_epoch.size(); }
			else { num_examples_in_batch = mini_batch_size; }

			// Calculate all gradients for this example + total cost
			// Indexing here is a little weird:
			//		The 0th entry corresponds to the 0th activation vec returned by NeuralNetwork,
			//		which corresponds to the weights connecting the 1st layer to the 0th layer.
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
				calcGradients(*weighted_inputs, *activations, weight_grad_loc, bias_grad_loc);
				for(int layer = 0; layer < weight_grad_loc->size(); layer++) {
					if(batch == 0) {
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
			BOOST_LOG_TRIVIAL(info) << "Cost before this epoch: " << cost;

			// Update weight/bias on each layer
			for(int layer = 0; layer < weight_grad.size(); layer++) {
				// Add 1 to layer when updating (b/c of weird indexing, see earlier comment).
				// 0th weight grad matrix represents connection from 1st layer to 0th layer,
				// so we need to pass 1 here.
				int layerCanonical = layer + 1;

				// Calc avg for SGD
				weight_grad[layer] /= num_examples_in_batch;
				bias_grad[layer] /= num_examples_in_batch;

				// Update network
				network->setLayerProperties(
					layerCanonical,
					network->getWeights(layerCanonical) - (weight_grad[layer] * learning_rate),
					network->getBiases(layerCanonical) - (bias_grad[layer] * learning_rate)
				);
			}
		}
	}
}
