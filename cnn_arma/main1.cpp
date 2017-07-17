
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include "armadillo"
#include <string>
#include "Neural_Net.h"

using namespace std;
using namespace arma;

#define TRAINING_SIZE 60
//#define TRAINING_TEST 10000

double sigmoid(double in)
{
		return (1/(1+exp(-in)));
}

double der_sigmoid(double in)
{
		return in*(1-in);
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

void read_Mnist2(string filename, int n_train_file, arma::cube (&cube_in),int &row, int &col)
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
        
        row = n_rows;
        col = n_cols;

        cube_in.resize(n_rows,n_cols,n_train_file);
        cube_in.fill(0);
 
        for(int i = 0; i < n_train_file; ++i)
        {
            for(int r = 0; r < n_rows; ++r){
	            for(int c = 0; c < n_cols; ++c)
	            {
	                    unsigned char temp = 0;
	                    file.read((char*) &temp, sizeof(temp));
	                    temp = (double) temp;
	                    
	                    cube_in.at(r,c,i)=(double)temp;
	                    //cube_in.at(r,c,i) = temp<127?1:0;

	            }
        	}
        }
    }
}


arma::vec number_to_vec(int n){
	arma::vec vec_out;
	if( n<0 && n>9){
		printf("error: n no esta entre \"0<=n<=9\" \n");
		return vec_out;
	}
	vec_out.resize(10);
	vec_out.fill(0);
	vec_out.at(n) = 1;
	return vec_out;
}

void read_Mnist_Label2(string filename, arma::cube (&cube_in), int n_train_file)
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

        //printf("%d\n",magic_number );
        //printf("%d\n",number_of_images );
        cube_in.resize(1,10,n_train_file);
        cube_in.fill(0);

        //for(int i = 0; i < number_of_images; ++i)
        for(int i = 0; i < n_train_file; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            cube_in.at(0,(int)temp,i)=1;
            //vec.at(i)= (double)temp;
            //printf("<<<%f\n", (double)temp );
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


void conv2(arma::mat (&in), int n, int m, arma::mat (&conv), int p, int q, arma::mat(&out)){
	if( n<=0 || m<=0 || p<=0 || q<=0 ){
		printf(">>>>> Error: matriz sin dimension <<<<<\n");
		return;
	}

	out.copy_size(in);
	out.fill(0);
	for(int i=0 ; i<n ; i++){
		for(int j=0 ; j<m ; j++){		
			double offset = 0;
			for(int k=0 ; k<p ; k++){
				for(int l=0 ; l<q ; l++){
					double cc = conv.at(k,l);
					double dd = 0;
					if( (i-(int)(p/2)+k)>=0 && (j-(int)(q/2)+l)>=0 &&
						(i-(int)(p/2)+k)<n && (j-(int)(q/2)+l)<m  ){
						dd = in.at( i-(int)(p/2)+k , j-(int)(q/2)+l );
					}
					offset += cc*dd;
				}
			}
			out.at(i,j) = offset>0? offset:0;
		}
	}
}

void max_pooling(arma::mat (&in), int n_rows, int n_cols, int tam_maxpool, arma::mat (&out),int fil, int col){
	if( n_rows<tam_maxpool && n_cols<tam_maxpool && tam_maxpool<=1){
		printf(">>>>>> Error: max_pooling mayor que matriz <<<<<\n");
		return;
	}
	
	//int fil = (n_rows%2==0)?(n_rows/tam_maxpool):((n_rows+1)/tam_maxpool);
	//int col = (n_cols%2==0)?(n_cols/tam_maxpool):((n_cols+1)/tam_maxpool);

	out.set_size(fil,col);	
	out.fill(0);


	for(int i=0 ; i<fil; i++){
		for(int j=0 ; j<col ; j++){
			double max = in.at(i*tam_maxpool,j*tam_maxpool);
			for(int k=i*tam_maxpool ; k<(i+1)*tam_maxpool ; k++){
				for(int l=j*tam_maxpool ; l<(j+1)*tam_maxpool ; l++){
					if( k<n_rows && l<n_cols ){
						double temp = in.at(k,l);
						if( max<temp ){
							max = temp;
						}
					}
				}
			}
			out.at(i,j)=max;
		}
	}
}



void print_cube(arma::cube (&cube_in)){
	for(int k=0;k<TRAINING_SIZE;k++){
		for(int i=0;i<28;i++){
			for(int j=0;j<28;j++){
				cout<<cube_in.at(i,j,k) <<" "; 
			}
			cout<<endl;
		}
		cout<<endl;
		cout<<endl;
	}
}


void print_mat(string message,arma::mat (&mat_in),int n_rows, int n_cols){
	cout<< message <<": "<<endl;
	for(int i=0;i<n_rows;i++){
    	for(int j=0;j<n_cols;j++){
    		printf("%0.0f ",mat_in.at(i,j));
    	}
    	printf("\n");
    }
}

void mat_rand(arma::mat (&in),int n_rows, int n_cols){
	int min = -5;
	int max = 5;
	//srand(time(NULL));
	for(int i=0 ; i<n_rows ; i++){
		for(int j=0 ; j<n_cols ; j++){
			in.at(i,j) = (rand()%(max-min))+min;
		}
	}

}

arma::cube conv_relu_pooling(arma::cube (&cube_in), int &n_rows, int &n_cols, int n_slice, int n_maxPool, int tam_kernelConv){
	int p = tam_kernelConv;
	int q = tam_kernelConv;
	// cambiar por un random hasta negativos
		//arma::mat mat_kenelConv = randu<mat>(p,q); 
	arma::mat mat_kenelConv = arma::zeros<arma::mat>(p,q); 
	mat_rand(mat_kenelConv,p,q);
	//mat_kenelConv.print("fffffffffffffffffffffffffff");

	arma::cube cube_conv;
	cube_conv.resize(n_rows,n_cols,n_slice);
	cube_conv.fill(0);
	
	for(int i=0 ; i<n_slice ; i++){
		mat temp_conv;
		conv2(cube_in.slice(i),n_rows,n_cols,mat_kenelConv,p,q,temp_conv);
		cube_conv.slice(i) = temp_conv;
	}
	
	int temp_n_rows = (n_rows%2==0)?(n_rows/n_maxPool):((n_rows+1)/n_maxPool);
	int temp_n_cols = (n_cols%2==0)?(n_cols/n_maxPool):((n_cols+1)/n_maxPool);	
		
	arma::cube cube_maxPool;
	cube_maxPool.resize(temp_n_rows,temp_n_cols,n_slice);
	cube_maxPool.fill(0);

	for(int j=0 ; j<n_slice ; j++){
		mat temp_maxpooling;
		max_pooling(cube_conv.slice(j),n_rows,n_cols,n_maxPool,temp_maxpooling,temp_n_rows,temp_n_cols);		
		cube_maxPool.slice(j) = temp_maxpooling; 
	}
	
	n_rows = temp_n_rows;
	n_cols = temp_n_cols;

	return cube_maxPool;
}

void my_vectorise_mat(arma::mat (&mat_in), int n_rows, int n_cols, arma::vec (&vec_out), int &n_out ){
	//vec_out = vectorise(mat_in);
	n_out = n_rows*n_cols;
	vec_out.resize(n_out);
	for(int i=0 ; i<n_rows ; ++i){
		for(int j=0 ; j<n_cols ; ++j){
			vec_out.at(j + i*n_cols) = mat_in.at(i,j);
		}
	}
}


void my_vectorise_cube(arma::cube (&cube_in), int n_rows, int n_cols, int n_slices, arma::vec (&vec_out), int (&n_out) ){
	n_out = n_rows*n_cols*n_slices;
	vec_out.resize(n_out);
	for(int k=0 ; k<n_slices ; k++){
		for(int i=0 ; i<n_rows ; ++i){
			for(int j=0 ; j<n_cols ; ++j){
				vec_out.at(j + i*n_cols + k*(n_cols*n_rows)) = cube_in.at(i,j,k);
			}
		}
	}
}




class cnn{
	int _tam_kernelConv;
	int _n_maxPool;
	int _slices_per_img;

    int _n_rows; 
    int _n_cols;
	arma::cube _cube_training;
	arma::cube _cube_labels;

	Neural_Net my_network;
	int _n_train_file;

	vector< vec > training_in;
	vector< vec > training_out;

public:

	cnn(int tam_kernelConv, int n_maxPool, int slices_per_img){
		_tam_kernelConv = tam_kernelConv;
		_n_maxPool 		= n_maxPool;
		_slices_per_img = slices_per_img;
		_n_rows = 0; 
    	_n_cols = 0;

    	
    	int vec_features = 24;
		int vec_output = 10;

		vector<int> neurons;
		neurons.push_back(vec_features);
		neurons.push_back(vec_features);
		neurons.push_back(5);
		neurons.push_back(vec_output);

		my_network.init(4,neurons);
	}

	void init(string train_file, string labels_file, int n_train_file){
		_n_train_file = n_train_file;


		//read_Mnist_Label2(labels_file, _cube_labels, _n_train_file);
		read_Mnist_Label(labels_file, training_out);


		//cube_labels.print("cube labels:"); 

	    read_Mnist2(train_file,_n_train_file,_cube_training,_n_rows,_n_cols);
	    //printf("_n_rows:%d , _n_cols:%d\n",_n_rows,_n_cols );

	}

	arma::vec feature_extraction(arma::mat mat_img){
		arma:cube cube_temp;
			cube_temp.resize(_n_rows,_n_cols,_slices_per_img);
			cube_temp.fill(0);
			int temp_n_rows = _n_rows;
			int temp_n_cols = _n_cols;
		for(int i=0 ; i<_slices_per_img ; i++){
			cube_temp.slice(i) = mat_img;
		}

		arma::cube cube_out;	

		int min = 2; //2x2
		while(temp_n_cols>min && temp_n_rows>min){
			cube_out = conv_relu_pooling(cube_temp, temp_n_rows, temp_n_cols, _slices_per_img, _n_maxPool, _tam_kernelConv);
			cube_temp = cube_out;
		}	


		arma::vec vec_out;
		int n_vec = 0;
		my_vectorise_cube(cube_out, temp_n_rows, temp_n_cols, _slices_per_img, vec_out, n_vec);
		//vec_out.print("final");
		return vec_out;
	}

	void clasification(){
		// vec es la entrada para la red neuronal
		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>AQUI<<<<<<<<<<<<<<<<<<<<<<<<<<
		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>AQUI<<<<<<<<<<<<<<<<<<<<<<<<<<
		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>AQUI<<<<<<<<<<<<<<<<<<<<<<<<<<
		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>AQUI<<<<<<<<<<<<<<<<<<<<<<<<<<
		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>AQUI<<<<<<<<<<<<<<<<<<<<<<<<<<

		

		my_network.get_training_set(training_in,training_out);
		my_network.set_activation_function(sigmoid,der_sigmoid);
		my_network.set_learning_rate(0.2);
		//Definiendo el momemtum - para ignorar este parametro basta con utilizar un momemtum = 0
		my_network.set_momentum(0.1);
		//my_network.backpropagation_sgd(1000);
		my_network.backpropagation_mini_batch(1000,100);
	}

	void training(){

		arma::vec vec_features;
		arma::vec vec_label;
		for(int i=0 ; i<_n_train_file ; i++){	
			//vec_label 	 = vectorise(_cube_labels.slice(i));
			//printf("i de for: %d\n",i);
			//vec_label.print("label");
			vec_features = feature_extraction(_cube_training.slice(i));
			training_in.push_back(vec_features);
		}
		//clasification();
		training_in[5].print("zzz");
		training_out[5].print("111111");
		
	}


};




int main(){
	
	string train_file  = "dataset/train-images-idx3-ubyte";
    string labels_file = "dataset/train-labels-idx1-ubyte";

	
	int n_maxPool = 2; 		//2x2
	int tam_kernelConv = 3; //3x3
	int slices_per_img = 6; //6x6 

   	cnn my_cnn(tam_kernelConv, n_maxPool, slices_per_img);
	my_cnn.init(train_file,labels_file,TRAINING_SIZE);    
	my_cnn.training();
	


	return 0;
	
}
