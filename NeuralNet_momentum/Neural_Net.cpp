#include "Neural_Net.h"

Neural_Net::Neural_Net(int num_layers,vector<int> num_neurons)
{
		num_of_layers = num_layers;
		num_of_neurons = num_neurons;

		for(int i=0;i<num_of_layers-1;i++)
		{
			Layer tmp(num_of_neurons[i]);
			tmp.set_weights(num_of_neurons[i+1],num_of_neurons[i]+1);
			tmp.set_prev_gradient(num_of_neurons[i+1],num_of_neurons[i]+1);
			layers.push_back(tmp);
		}

		Layer my_tmp(num_of_neurons[num_of_layers-1]);
		layers.push_back(my_tmp);

}

void Neural_Net::get_training_set(vector<vec> training_in,vector<vec> training_out)
{
		training_input = training_in;
		training_output = training_out;
}


void Neural_Net::set_activation_function(double (*f)(double), double (*der_f)(double))
{
		activation_function = f;
		der_activation_function = der_f;
}

void Neural_Net::set_learning_rate(double l)
{
		learning_rate = l;
}

void Neural_Net::set_momentum(double m)
{
		momentum = m;
}

vec Neural_Net::apply_func(vec my_vec,double (*f)(double))
{
		vec tmp;
		int vec_size = my_vec.n_elem;
		tmp.zeros(vec_size);
		for(int i=0;i<vec_size;i++)
		{
			tmp(i) = f(my_vec(i));
		}
		return tmp;

}

void Neural_Net::forward_propagate(int j)
{
		layers[0].set_activation_units(training_input[j]);
		layers[0].set_bias_coefficient();
		vec tmp_result;

		for(int i=1;i<num_of_layers-1;i++)
		{
			tmp_result = layers[i-1].weights*layers[i-1].activation_units;
			tmp_result = apply_func(tmp_result,activation_function);
			layers[i].set_activation_units(tmp_result);
			layers[i].set_bias_coefficient();
		}

		tmp_result = layers[num_of_layers-2].weights*layers[num_of_layers-2].activation_units;
		tmp_result = apply_func(tmp_result,activation_function);
		layers[num_of_layers-1].set_activation_units(tmp_result);
}


void Neural_Net::test_propagate(int j)
{
		layers[0].set_activation_units(test_input[j]);
		layers[0].set_bias_coefficient();
		vec tmp_result;

		for(int i=1;i<num_of_layers-1;i++)
		{
			tmp_result = layers[i-1].weights*layers[i-1].activation_units;
			tmp_result = apply_func(tmp_result,activation_function);
			layers[i].set_activation_units(tmp_result);
			layers[i].set_bias_coefficient();
		}

		tmp_result = layers[num_of_layers-2].weights*layers[num_of_layers-2].activation_units;
		tmp_result = apply_func(tmp_result,activation_function);
		layers[num_of_layers-1].set_activation_units(tmp_result);
}

void Neural_Net::predict(vector<vec> t_i)
{
		test_input = t_i;

		for(int i=0;i<test_input.size();i++)
		{
				test_propagate(i);
				test_input[i].print();
				cout<<endl;
				layers[num_of_layers-1].activation_units.print();
				cout<<endl;
		}

}

void Neural_Net::back_propagate(int j)
{
		int layers_size = layers.size();
		
		vec tmp_der,tmp_delta = layers[layers_size-1].activation_units - training_output[j];
		layers[layers_size-1].set_deltas(tmp_delta);
		mat tmp_gradient;

		for(int i = layers_size-2;i>=1;i--)
		{
			tmp_der = apply_func(layers[i].activation_units,der_activation_function);
			tmp_delta = layers[i].weights.t()*layers[i+1].deltas;
			tmp_delta = tmp_delta % (tmp_der);
			tmp_delta.shed_row(0);
			layers[i].set_deltas(tmp_delta);
		}

		for(int k=0;k<layers_size-1;k++)
		{
			tmp_gradient = (-1.0)*(layers[k+1].deltas*(layers[k].activation_units.t()));
			layers[k].weights = layers[k].weights + learning_rate*tmp_gradient + momentum*layers[k].prev_gradient;
			layers[k].prev_gradient = tmp_gradient;
		}
}

void Neural_Net::backpropagation(int iters)
{
		double err=0;
		int t = training_input.size();
		for(int k=0;k<iters;k++)
		{
			for(int i=0;i<t;i++)
			{	
				forward_propagate(i);
				err += out_error(i);
				back_propagate(i);
			}
			cout<<"Iter "<<k<<" Error: "<<err<<endl;
			err=0;
		}	
}

void Neural_Net::backpropagation_sgd(int iters)
{
		double err;
		int t = training_input.size();
		for(int k=0;k<iters;k++)
		{
			
				forward_propagate(k%t);
				err = out_error(k%t);
				back_propagate(k%t);

			cout<<"Iter "<<k<<" Error: "<<err<<endl;
		}	
}

void Neural_Net::backpropagation()
{
		double err=0.5;
		int my_iters =0;

		while(err>0.01)
		{
			err = 0;
			for(int i=0;i<training_input.size();i++)
			{	
				forward_propagate(i);
				err += out_error(i);
				back_propagate(i);
			}
			my_iters++;

		}
		cout<<"Iters: "<<my_iters<<endl;
}

void Neural_Net::backpropagation_sgd()
{
		double err=0.5;
		int my_iters =0;
		int t = training_input.size();

		while(err>0.00000001)
		{
			forward_propagate(my_iters%t);
			err = out_error(my_iters%t);
			back_propagate(my_iters%t);
			my_iters++;

		}
		cout<<"Iters: "<<my_iters<<endl;
}

double Neural_Net::out_error(int k)
{
		double sum_error = 0;
		int layers_size = layers.size();
		
		vec tmp_vector_error = layers[layers_size-1].activation_units - training_output[k];
        
		for(int i=0;i<tmp_vector_error.n_elem;i++)
		{
			sum_error += pow(tmp_vector_error(i),2);
		}

		sum_error *= 0.5;
		return sum_error;
}
