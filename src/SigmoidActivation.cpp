/*
 * SigmoidActivation.cpp
 *
 *  Created on: Feb 16, 2018
 *      Author: danielruskin
 */

#include "SigmoidActivation.h"

std::unique_ptr<arma::colvec> SigmoidActivation::eval(const arma::colvec& input) {
	std::unique_ptr<arma::colvec> ret;
	ret.reset(new arma::colvec(1 / (1 + arma::exp(-1 * input))));
	return std::move(ret);
}

std::unique_ptr<arma::colvec> SigmoidActivation::evalPrime(const arma::colvec& input) {
	std::unique_ptr<arma::colvec> sigmoidVal = eval(input);

	std::unique_ptr<arma::colvec> ret;
	ret.reset(new arma::colvec((*sigmoidVal) * (1 - (*sigmoidVal))));
	return std::move(ret);
}
