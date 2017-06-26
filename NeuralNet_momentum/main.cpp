#include <iostream>
#include <fstream>
#include <cstdlib>
#include "Neural_Net.h"
using namespace std;

double sigmoid(double in)
{
		return (1/(1+exp(-in)));
}

double der_sigmoid(double in)
{
		return in*(1-in);
}

void read_dataset(string file, vector<vec>& t_in, vector<vec>& t_out, int data_s, int input_s, int output_s)
{
	ifstream my_file;
	my_file.open(file.c_str());

	double tmp_data;
	vec tmp_vec1(input_s), tmp_vec2(output_s);

	for(int i=0;i<data_s;i++)
	{	
		for(int j=0;j<input_s;j++)
		{
			my_file>>tmp_data;
			switch(j)
			{
				case 0: tmp_data = tmp_data/7.9;
						break;
				case 1: tmp_data = tmp_data/4.4;
						break;
				case 2: tmp_data = tmp_data/6.9;
						break;
				case 3: tmp_data = tmp_data/2.5;
						break;
			}
			tmp_vec1(j) = tmp_data;
		}
		t_in.push_back(tmp_vec1);

		for(int k=0;k<output_s;k++)
		{
			my_file>>tmp_vec2(k);
		}
		t_out.push_back(tmp_vec2);
	}
}

int main()
{
	string name_file = "irisData.txt";
	string test_name_file = "irisDataTest.txt";
	vector<vec> in_iris;
	vector<vec> out_iris;

	vector<vec> test_in_iris;
	vector<vec> test_out_iris;


	read_dataset(name_file,in_iris,out_iris,150,4,3);
	read_dataset(name_file,test_in_iris,test_out_iris,150,4,3);

	/*
	vector<vec> train;
	vec tmp;
 	//Primer tipo
	tmp<<5.1/7.9<<3.5/4.4<<1.4/6.9<<0.2/2.5;
	train.push_back(tmp); 
	tmp<<4.9/7.9<<3.0/4.4<<1.4/6.9<<0.2/2.5;
	train.push_back(tmp); 
	tmp<<4.7/7.9<<3.2/4.4<<1.3/6.9<<0.2/2.5;
	train.push_back(tmp);

	// Segundo tipo
	tmp<<7.0/7.9<<3.2/4.4<<4.7/6.9<<1.4/2.5;
	train.push_back(tmp);
	tmp<<6.4/7.9<<3.2/4.4<<4.5/6.9<<1.5/2.5;
	train.push_back(tmp);
	tmp<<6.9/7.9<<3.1/4.4<<4.9/6.9<<1.5/2.5;
	train.push_back(tmp);

	// Tercer tipo
	tmp<<6.3/7.9<<3.3/4.4<<6.0/6.9<<2.5/2.5;
	train.push_back(tmp);
	tmp<<5.8/7.9<<2.7/4.4<<5.1/6.9<<1.9/2.5;
	train.push_back(tmp);
	tmp<<7.1/7.9<<3.0/4.4<<5.9/6.9<<2.1/2.5;
	train.push_back(tmp);*/

	vector<int> neurons;
	neurons.push_back(4);
	neurons.push_back(3);
	neurons.push_back(3);

	Neural_Net my_network(3,neurons);

	my_network.get_training_set(in_iris,out_iris);
	my_network.set_activation_function(sigmoid,der_sigmoid);
	my_network.set_learning_rate(0.2);
	//Definiendo el momemtum - para ignorar este parametro basta con utilizar un momemtum = 0
	my_network.set_momentum(0.1);
	my_network.backpropagation_sgd();
	//my_network.backpropagation();

	my_network.rate(test_in_iris,test_out_iris);

	//my_network.predict(train);
	return 0;
}
