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
#include "SigmoidActivation.h"
#include "MeanSquaredErrorCost.h"

//! Program Entrypoint
int main(int argc, char **argv)
{
	arma::arma_rng::set_seed_random();

	VecOfInts config;
	config.push_back(2);
	config.push_back(2);
	NeuralNetwork<SigmoidActivation, MeanSquaredErrorCost> nn(config);

	arma::colvec input(2);
	input(0) = 1;
	input(1) = 0.5;
	std::unique_ptr<arma::colvec> val = nn.predict(input);
}
