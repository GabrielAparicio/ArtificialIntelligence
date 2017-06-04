#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include "Back_Propagation.h"

using namespace std;


#define TRAINING_SIZE 60000
#define INPUT_SIZE 28*28

#define TRAINING_TEST 10000

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


void read_Mnist2(string filename, double** vec)
{
    //ifstream file (filename, ios::binary);
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
        
        for(int i = 0; i < TRAINING_SIZE; ++i)
        {
            for(int r = 0; r < n_rows*n_cols; ++r)
            {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    temp = (double) temp;
                    
                    if(temp>127)
                    	vec[i][r] = 1;
                    else
                    	vec[i][r] = 0;
                    //vec[i][r]=(double)temp;
            }
        }
    }
}

void read_Mnist_Label2(string filename, double* vec)
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

        //for(int i = 0; i < number_of_images; ++i)
        for(int i = 0; i < TRAINING_SIZE; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (double)temp;
        }
    }
}




void create(double** &vec, int n, int m){
    vec = new double*[n];
    for(int i=0;i<n;i++){
        vec[i]=new double[m];
    }
}

void fill(double** vec, int n, int m, int value=0){
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            vec[i][j] = value;
        }
    }
}


void print_matrix(double** vec, int n, int m){
    for(int i=0;i<2;i++){
        for(int j=0;j<m;j++){
            cout<< vec[i][j] <<" ";
        }
        cout<<endl;
    }
}



void print_vec(double* vec, int n){
    for(int i=0;i<n;i++){
        cout<<vec[i]<<" ";
    }
    cout<<endl;
}

void print_matrix_image(double** vec,int k)
{
	for(int i=0;i<28;i++)
	{
		for(int j=0;j<28;j++)
		{
			cout<<vec[k][i*28+j]<<" ";

		}
		cout<<endl;
	}


	for(int i=784;i<794;i++)
	{
		cout<<vec[k][i]<<" ";
	}
	cout<<endl;

}

void print_vec(double** vec, int k){
    for(int i=0;i<10;i++){
        cout<<vec[k][i]<<" ";
    }
    cout<<endl;
}

double error_vec(double* vec1, double* vec2, int n){
	double err=0;
	for(int i=0;i<n;i++){
		err+=pow(vec1[i]-vec2[i],2);
	}
	return  err/2;
}



int main(int argc, char* argv[]){

	string train_file  = "train-images-idx3-ubyte";
    string labels_file = "train-labels-idx1-ubyte";

    int number_of_images = TRAINING_SIZE;
    int image_size = INPUT_SIZE; 
  
    double** entradas=NULL;
    create(entradas,number_of_images,image_size+10);
    fill(entradas,number_of_images,image_size+10,0);
    read_Mnist2(train_file, entradas);

    double* labels_vec=NULL;
    labels_vec = new double[number_of_images];
    read_Mnist_Label2(labels_file, labels_vec);

    for(int i=0;i<number_of_images;i++){
        int j=image_size+labels_vec[i];
        entradas[i][j]=1;
    }   
    cout<<"---------------------------------"<<endl;


    cout<<"$$$$$$$$$$$$$$$$"<<endl;
	string train_file_test  = "t10k-images-idx3-ubyte";
    string labels_file_test = "t10k-labels-idx1-ubyte";
	double** testData = NULL;
    	create(testData,number_of_images,image_size);
    	fill(entradas,number_of_images,image_size,0);
    read_Mnist2(train_file_test, testData);
    
	double** labels_vec_bin=NULL;
    	create(labels_vec_bin,number_of_images,10);
    	fill(labels_vec_bin,number_of_images,10,0);


    double* labels_vec_testData = NULL;
    labels_vec_testData = new double[number_of_images];
    read_Mnist_Label2(labels_file_test, labels_vec_testData);
   
    for(int i=0;i<number_of_images;i++){
        int j=labels_vec_testData[i];
        labels_vec_bin[i][j]=1;
    }
    
    cout<<"---------------------------------"<<endl;
    
    int numLayers = 6 ,lSz[6] = {INPUT_SIZE,15,15,15,15,10};
	double beta = 0.3, alpha = 0.1, thresh =  0.00001;
	long num_iter = 50000;

	Back_Propagation *bp = new Back_Propagation(numLayers, lSz, beta, alpha);

	cout<<  "Entrenando...." << endl;

    long i;

	for (i=0; i<num_iter ; i++){
		bp->bpgt(entradas[i%TRAINING_SIZE], &entradas[i%TRAINING_SIZE][INPUT_SIZE]);

		if( bp->mse(&entradas[i%TRAINING_SIZE][INPUT_SIZE]) < thresh) {
			cout << endl << "Red entrenada en " << i << " iteraciones." << endl;
			cout << "Error:  " << bp->mse(&entradas[i%TRAINING_SIZE][INPUT_SIZE])
				 <<  endl <<  endl;
			break;
		}
	}
	//double xxx=0;
	if ( i == num_iter )
		//xxx = bp->mse(&entradas[(i-1)%TRAINING_SIZE][INPUT_SIZE]);
		cout << endl << i << " Numero maximo de iteraciones alcanzado..."
		<< "Error: " << bp->mse(&entradas[(i-1)%TRAINING_SIZE][INPUT_SIZE]) << endl;
	
	int count=0;
	for ( i = 0 ; i < TRAINING_TEST ; i++ ){

		bp->ffwd(testData[i]);
		double* temp=new double[10];
		for(int k=0;k<10;k++){
			temp[k]=bp->Out(k);
		}
		
		double error= error_vec(labels_vec_bin[i],temp,10);
        //if(error<=xxx){
	    if(error<=0.06){
	    	count++	;
	    }

	}
	cout<<"rate :"<< (double)count/(double)TRAINING_TEST*100<<" %"<<endl;
	
	return 0;
}
