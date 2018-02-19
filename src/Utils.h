/*
 * Utils.h
 *
 *  Created on: Feb 14, 2018
 *      Author: danielruskin
 */

#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <vector>
#include <armadillo>

typedef std::vector<arma::mat> VecOfMats;
typedef std::vector<arma::colvec> VecOfColVecs;
typedef std::vector<int> VecOfInts;
typedef std::pair<arma::colvec, arma::colvec> TrainingExample; // Pair of <feature, correct_prediction>
typedef std::vector<TrainingExample> TrainingExamples;

#endif /* SRC_UTILS_H_ */
