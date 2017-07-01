#include "Layer.h"

Layer::Layer(int n_neurons)
{
		activation_units.zeros(n_neurons);
		deltas.zeros(n_neurons);
		num_neurons = n_neurons;
}

void Layer::reset_delta()
{
	deltas.zeros(num_neurons);
}

void Layer::set_bias_coefficient()
{
		activation_units.insert_rows(0,1);
		activation_units(0) = 1.0;
}

void Layer::set_num_neurons(int num_neurons)
{
		activation_units.zeros(num_neurons);
}

void Layer::set_activation_units(vec activation)
{
		activation_units = activation;
}

void Layer::set_weights(int dimx,int dimy)
{
		weights.randu(dimx,dimy);
}

void Layer::set_prev_gradient(int dimx,int dimy)
{
		prev_gradient.zeros(dimx,dimy);
}

void Layer::set_new_weight(mat new_weight)
{
		weights = new_weight;
}

void Layer::set_deltas(vec del)
{
		deltas = del;
}