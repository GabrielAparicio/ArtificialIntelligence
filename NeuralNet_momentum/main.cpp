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

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


void read_Mnist(string filename, vector<vec>& vec_in)
{
    ifstream file(filename.c_str(),ios::binary);
    
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        unsigned char temp = 0;
        vec vec_tmp(n_rows*n_cols);

        
        for(int i = 0; i < number_of_images; ++i)
        {
            for(int r = 0; r < n_rows*n_cols; ++r)
            {       
                    file.read((char*) &temp, sizeof(temp));
                    temp = (double) temp;

                    if(temp>127)
                    	vec_tmp(r)= 1;
                    else
                    	vec_tmp(r) = 0;
                    
            }
            vec_in.push_back(vec_tmp);
        }
    }
}


void read_Mnist_Label(string filename, vector<vec>& vec_in)
{
    ifstream file(filename.c_str(),ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        vec vec_tmp;

        for(int i = 0; i < number_of_images; ++i)
        {
        	vec_tmp.zeros(10);
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));

            vec_tmp((int)temp) = 1.0;
            vec_in.push_back(vec_tmp);
            //vec[i]= (double)temp;
        }
    }
}

int main()
{
    //string name_file = "irisData.txt";
    //string test_name_file = "irisDataTest.txt";

    string name_file = "irisData90.txt";
    string test_name_file = "testData60.txt";

    vector<vec> in_iris;
    vector<vec> out_iris;

    vector<vec> test_in_iris;
    vector<vec> test_out_iris;

    //read_dataset(name_file,in_iris,out_iris,150,4,3);
    //read_dataset(name_file,test_in_iris,test_out_iris,150,4,3);

    read_dataset(name_file,in_iris,out_iris,90,4,3);
    read_dataset(name_file,test_in_iris,test_out_iris,60,4,3);

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

    my_network.backpropagation();
    //my_network.backpropagation_mini_batch(30);
	//my_network.backpropagation_sgd();

	my_network.rate(test_in_iris,test_out_iris);
	return 0;
}