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
	boost::log::add_common_attributes();

	srand(time(0));
	arma::arma_rng::set_seed_random();

	VecOfInts config;
	config.push_back(2);
	config.push_back(2);
	std::shared_ptr<NeuralNetwork<SigmoidActivation, MeanSquaredErrorCost>> nnp;
	nnp.reset(new NeuralNetwork<SigmoidActivation, MeanSquaredErrorCost>(config));

	std::shared_ptr<const TrainingExamples> ex;
	NeuralNetworkTrainer<SigmoidActivation, MeanSquaredErrorCost> nnt(
		nnp,
		ex,
		0.1,
		10,
		10
	);
	nnt.trainNetwork();
}
