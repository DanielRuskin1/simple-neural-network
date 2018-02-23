/*
 * Main.cpp
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#include <armadillo>
#include <iostream>
#include "NeuralNetwork.h"
#include "NeuralNetwork.cpp"
#include "NeuralNetworkTrainer.h"
#include "NeuralNetworkTrainer.cpp"
#include "SigmoidActivation.h"
#include "MeanSquaredErrorCost.h"

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

//! Program Entrypoint
int main(int argc, char **argv)
{
	// Setup logging/random
	boost::log::add_common_attributes();
	srand(time(0));
	arma::arma_rng::set_seed_random();

	// Setup NN
	BOOST_LOG_TRIVIAL(debug) << "Setting up neural network...";
	VecOfInts config;
	config.push_back(784);
	config.push_back(30);
	config.push_back(10);
	std::shared_ptr<NeuralNetwork<SigmoidActivation, MeanSquaredErrorCost>> nnp;
	nnp.reset(new NeuralNetwork<SigmoidActivation, MeanSquaredErrorCost>(config));

	// Load training data
	BOOST_LOG_TRIVIAL(debug) << "Loading training data...";
	std::shared_ptr<TrainingExamples> ex(new TrainingExamples);

	arma::mat features;
	features.load("/Users/danielruskin/Downloads/mnist_train.csv");
	arma::colvec labels = features.col(0);
	features.shed_col(0);

	for(unsigned int row = 0; row < features.n_rows; row++) {
		TrainingExample te;
		te.first = arma::trans(features.row(row));
		te.second.resize(10);
		for(int i = 0; i < 10; i++) {
			if(labels(row) == i) {
				te.second(i) = 1;
			} else {
				te.second(i) = 0;
			}
		}
		ex->push_back(te);
	}
	BOOST_LOG_TRIVIAL(debug) << "Loaded training data!  Samples: " << ex->size();

	BOOST_LOG_TRIVIAL(debug) << "Training neural network...";
	NeuralNetworkTrainer<SigmoidActivation, MeanSquaredErrorCost> nnt(
		nnp,
		ex,
		3.0,
		10,
		30
	);

	// Train network
	TrainingExample te = *(ex->begin());
	std::unique_ptr<arma::colvec> before = nnp->predict(te.first);
	nnt.trainNetwork();
	std::unique_ptr<arma::colvec> after = nnp->predict(te.first);

	// Simple test
	BOOST_LOG_TRIVIAL(debug) << "Trained neural network!";
	BOOST_LOG_TRIVIAL(info) << "Sample Prediction, Before: ";
	before->print();
	BOOST_LOG_TRIVIAL(info) << "Sample Prediction, After: ";
	after->print();
	BOOST_LOG_TRIVIAL(info) << "Sample Prediction, Correct: ";
	te.second.print();
}
