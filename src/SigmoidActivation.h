/*
 * SigmoidActivation.h
 *
 *  Created on: Feb 16, 2018
 *      Author: danielruskin
 */

#ifndef SRC_SIGMOIDACTIVATION_H_
#define SRC_SIGMOIDACTIVATION_H_

#include <armadillo>

class SigmoidActivation {
public:
	static std::unique_ptr<arma::colvec> eval(const arma::colvec& input);
	static std::unique_ptr<arma::colvec> evalPrime(const arma::colvec& input);
};

#endif /* SRC_SIGMOIDACTIVATION_H_ */
