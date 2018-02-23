/*
 * MeanSquaredErrorCost.cpp
 *
 *  Created on: Feb 16, 2018
 *      Author: danielruskin
 */

#include "MeanSquaredErrorCost.h"

// Eval for single training example
double MeanSquaredErrorCost::eval(const arma::colvec& predict, const arma::colvec& correct) {
	return (1.0 / 2.0) * arma::accu(arma::square(correct - predict));
}

// Eval for single training example
std::unique_ptr<arma::colvec> MeanSquaredErrorCost::evalPrime(const arma::colvec& predict, const arma::colvec& correct) {
	std::unique_ptr<arma::colvec> ret;
	ret.reset(new arma::colvec(predict - correct));

	return std::move(ret);
}
