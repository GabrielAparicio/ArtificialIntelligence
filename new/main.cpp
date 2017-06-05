#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include "Back_Propagation.h"

#define TRAINIG_SIZE 90
#define INPUT_SIZE 4
#define TRAINING_TEST 60

using namespace std;

string filename = "IrisDataFinal.txt";
double entradas [TRAINIG_SIZE][INPUT_SIZE + 3];
double testData[60][INPUT_SIZE];
double labelData[60][3];

void cargar_datos1()
{
	int i=0;
	int j=0;
	string filas;
	ifstream infile;
	infile.open(filename.c_str());
	while(!infile.eof()) 
	{
		getline(infile,filas); 
		stringstream linea(filas);	
		string valor;
		while(getline(linea,valor,','))
		{   
		    entradas[i][j] = atof(valor.c_str());
			j++;
		}
		j=0;
		i++;	
	}
	infile.close();
}


void cargar_datos2(string my_file)
{
	double tmp;
	ifstream infile;
	infile.open(my_file.c_str());

	for(int i=0;i<60;i++)
	{
		for(int j=0;j<4;j++)
		{
			infile>>tmp;
			testData[i][j] = tmp;
		}

		for(int k=4;k<7;k++)
		{
			infile>>tmp;
			labelData[i][k-4] = tmp;
		}
	}
	infile.close();
	
}



void print_matrix()
{
	for(int i=0;i<TRAINIG_SIZE;i++)
	{
		for(int j=0;j<INPUT_SIZE+3;j++)
			cout<<entradas[i][j]<<" ";
		cout<<endl;
	}		
}



double error_vec(double* vec1, double* vec2, int n){
	double err=0;
	for(int i=0;i<n;i++){
		err+=pow(vec1[i]-vec2[i],2);
	}
	return  err/2;
}



int main(int argc, char* argv[])
{

	string test_file = "testDataFinal.txt";

    cargar_datos1();
    cargar_datos2(test_file);

    int numLayers = 4 ,lSz[4] = {INPUT_SIZE,8,10,3};

	double beta = 0.3, alpha = 0.1, thresh =  0.00001;

	long num_iter = 2000000;

	Back_Propagation *bp = new Back_Propagation(numLayers, lSz, beta, alpha);

	cout<<  "Entrenando...." << endl;

    long i;

	for (i=0; i<num_iter ; i++)
	{

		bp->bpgt(entradas[i%TRAINIG_SIZE], &entradas[i%TRAINIG_SIZE][INPUT_SIZE]);

		if( bp->mse(&entradas[i%TRAINIG_SIZE][INPUT_SIZE]) < thresh) {
			cout << endl << "Red entrenada en " << i << " iteraciones." << endl;
			cout << "Error:  " << bp->mse(&entradas[i%TRAINIG_SIZE][INPUT_SIZE])
				 <<  endl <<  endl;
			break;
		}
	}

	if ( i == num_iter )
		cout << endl << i << " Numero maximo de iteraciones alcanzado..."
		<< "Error: " << bp->mse(&entradas[(i-1)%TRAINIG_SIZE][INPUT_SIZE]) << endl;

	cout<< "Utilizando los pesos obtenidos para hacer predicciones...." << endl << endl;


	int count=0;
	for (i = 0 ; i < TRAINING_TEST ; i++ ){
		
		bp->ffwd(testData[i]);
		double* temp=new double[3];
		for(int k=0;k<3;k++){
			temp[k]=bp->Out(k);
		}
		double error= error_vec(labelData[i],temp,3);
	    if(error<=0.06){
	    	count++	;
	    }
	}
	cout<<"rate :"<< (double)count/(double)TRAINING_TEST*100<<" %"<<endl;
	



	return 0;
}
