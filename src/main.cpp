/*
 * Main.cpp
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#include <armadillo>
#include <iostream>
#include "NeuralNetwork.h"
#include "SigmoidActivation.h"
#include "MeanSquaredErrorCost.h"

//! Program Entrypoint
int main(int argc, char **argv)
{
	arma::arma_rng::set_seed_random();

	VecOfInts config;
	config.push_back(3);
	config.push_back(4);
	config.push_back(5);
	NeuralNetwork<SigmoidActivation, MeanSquaredErrorCost> nn(config);
}
