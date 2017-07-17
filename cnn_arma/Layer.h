#ifndef LAYER
#define LAYER

#include <armadillo>
using namespace arma;

class Layer{

public:

	vec activation_units;
	mat weights;
	mat prev_gradient;
	vec deltas;
	int num_neurons;

	Layer(int n_neurons);

	void set_bias_coefficient();
	
	void set_num_neurons(int num_neurons);

	void set_activation_units(vec activation);

	void set_weights(int dimx,int dimy);

	void set_prev_gradient(int dimx,int dimy);
	
	void set_new_weight(mat new_weight);
	
	void set_deltas(vec del);

	void reset_delta();
};
#endif 