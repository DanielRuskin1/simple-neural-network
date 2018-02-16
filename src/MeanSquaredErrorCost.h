/*
 * MeanSquaredErrorCost.h
 *
 *  Created on: Feb 16, 2018
 *      Author: danielruskin
 */

#ifndef SRC_MEANSQUAREDERRORCOST_H_
#define SRC_MEANSQUAREDERRORCOST_H_

#include <memory>
#include <armadillo>

class MeanSquaredErrorCost {
public:
	static std::unique_ptr<arma::vec> eval(const arma::colvec& predict, const arma::colvec& correct);
	static std::unique_ptr<arma::vec> evalPrime(const arma::colvec& predict, const arma::colvec& correct);
};

#endif /* SRC_MEANSQUAREDERRORCOST_H_ */
