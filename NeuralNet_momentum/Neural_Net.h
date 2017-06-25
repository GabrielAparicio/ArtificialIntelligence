#ifndef NEURAL_NET
#define NEURAL_NET

#include "Layer.h"
#include <iostream>
#include <armadillo>
#include <cmath>
#include <vector>

using namespace std;
using namespace arma;

class Neural_Net{

private:
	vector<Layer> layers;
	vector<vec> training_input;
	vector<vec> training_output;
	vector<vec> test_input;

	int num_of_layers;
	vector<int> num_of_neurons;

	double (*activation_function) (double);
	double (*der_activation_function) (double);
	double learning_rate;
	double momentum;

public:
	Neural_Net(int num_layers,vector<int> num_neurons);

	void get_training_set(vector<vec> training_in,vector<vec> training_out);
	
	void set_activation_function(double (*f)(double), double (*der_f)(double));
	
	void set_learning_rate(double l);
	
	void set_momentum(double m);
	
	vec apply_func(vec my_vec,double (*f)(double));

	void forward_propagate(int j);
	
	void test_propagate(int j);
	
	void predict(vector<vec> t_i);

	void back_propagate(int j);
	
	void backpropagation(int iters);

	void backpropagation_sgd(int iters);
	
	//Proceso de backpropagation normal
	void backpropagation();
	
	//Backpropagation estocastico - stochastic gradient descent
	void backpropagation_sgd();

	double out_error(int k);

};
#endif 
