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
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

//! Program Entrypoint
int main(int argc, char **argv)
{
	BOOST_LOG_TRIVIAL(info) << "Initializing...";

	// Setup logging/random
	boost::log::add_common_attributes();
	srand(time(0));
	arma::arma_rng::set_seed_random();

	// Retrieve program options
	boost::program_options::options_description desc("Required options");
	desc.add_options()
		("help", "produce help message")
	    ("layers", boost::program_options::value<std::string>(), "A string that represents the # of nodes for each layer, separated by spaces.  The first is the input layer, the last is the output layer.")
		("training_set_features_file", boost::program_options::value<std::string>(), "A file with the training data features.")
		("training_set_labels_file", boost::program_options::value<std::string>(), "A file with the training data labels.")
		("test_set_features_file", boost::program_options::value<std::string>(), "A file with the test data features.")
		("test_set_labels_file", boost::program_options::value<std::string>(), "A file with the test data labels.")
		("learning_rate", boost::program_options::value<double>(), "Learning rate.")
		("samples_per_epoch", boost::program_options::value<int>(), "Samples per epoch.")
		("num_epochs", boost::program_options::value<int>(), "Num epochs.")
	;
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);

	// Output help if needed
	if (vm.count("help")) {
	    BOOST_LOG_TRIVIAL(info) << "Program Help: ";
	    std::cout << desc << std::endl;
	    return 1;
	}

	BOOST_LOG_TRIVIAL(info) << "Creating neural network...";
	VecOfInts config;
	if(vm.count("layers")) {
		std::vector<std::string> v;
		boost::algorithm::split(v, vm["layers"].as<std::string>(), boost::algorithm::is_space());

		for(int i = 0; i < v.size(); i++) {
			try {
				config.push_back(boost::lexical_cast<int>(v[i]));
			}
			catch(const boost::bad_lexical_cast &)
	        {
				BOOST_LOG_TRIVIAL(error) << "Error parsing layers!  Did you pass them correctly?";
	            return 0;
	        }
		}
	} else {
		BOOST_LOG_TRIVIAL(error) << "Layers not set!";
		return 0;
	}
	std::shared_ptr<NeuralNetwork<SigmoidActivation, MeanSquaredErrorCost>> network;
	network.reset(new NeuralNetwork<SigmoidActivation, MeanSquaredErrorCost>(config));

	BOOST_LOG_TRIVIAL(info) << "Loading training data...";
	std::shared_ptr<arma::mat> training_features(new arma::mat);
	std::shared_ptr<arma::mat> training_labels(new arma::mat);
	if(vm.count("training_set_features_file") && vm.count("training_set_labels_file")) {
		training_features->load(vm["training_set_features_file"].as<std::string>(), arma::csv_ascii);
		training_labels->load(vm["training_set_labels_file"].as<std::string>(), arma::csv_ascii);
	} else {
		BOOST_LOG_TRIVIAL(error) << "Training data features or labels not set!";
		return 0;
	}

	BOOST_LOG_TRIVIAL(info) << "Loading testing data...";
	std::shared_ptr<arma::mat> test_features(new arma::mat);
	std::shared_ptr<arma::mat> test_labels(new arma::mat);
	if(vm.count("test_set_features_file") && vm.count("test_set_labels_file")) {
		test_features->load(vm["test_set_features_file"].as<std::string>(), arma::csv_ascii);
		test_labels->load(vm["test_set_labels_file"].as<std::string>(), arma::csv_ascii);
	} else {
		BOOST_LOG_TRIVIAL(error) << "Test data features or labels not set!";
		return 0;
	}

	BOOST_LOG_TRIVIAL(info) << "Loading numeric parameters...";
	double learning_rate;
	if(vm.count("learning_rate")) {
		learning_rate = vm["learning_rate"].as<double>();
	} else {
		BOOST_LOG_TRIVIAL(error) << "Learning rate not set!";
		return 0;
	}
	int samples_per_epoch;
	if(vm.count("samples_per_epoch")) {
		samples_per_epoch = vm["samples_per_epoch"].as<int>();
	} else {
		BOOST_LOG_TRIVIAL(error) << "Samples per epoch not set!";
		return 0;
	}
	int num_epochs;
	if(vm.count("num_epochs")) {
		num_epochs = vm["num_epochs"].as<int>();
	} else {
		BOOST_LOG_TRIVIAL(error) << "Num epochs not set!";
		return 0;
	}

	BOOST_LOG_TRIVIAL(info) << "Training network...";
	NeuralNetworkTrainer<SigmoidActivation, MeanSquaredErrorCost> trainer(
		network,
		training_features,
		training_labels,
		test_features,
		test_labels,
		learning_rate,
		samples_per_epoch,
		num_epochs
	);
	trainer.trainNetwork();

	BOOST_LOG_TRIVIAL(info) << "Done!";
}
