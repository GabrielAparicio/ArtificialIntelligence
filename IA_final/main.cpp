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
		//p[i][j]=1.0;
		
		while(getline(linea,valor,','))
		{   
		    entradas[i][j] = atof(valor.c_str());
			j++;
			//j=5 es la salida deseada
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


int main(int argc, char* argv[])
{
	
	///////////////////////////////////////////////////
	//float **entradas;
	string test_file = "testDataFinal.txt";

   cargar_datos1();
   cargar_datos2(test_file);
   //print_matrix();
    //cout<<endl;
/*
   	double entradas[TRAINIG_SIZE][INPUT_SIZE+3] = {
    	5.1,	3.5,	1.4,	0.2, 1.0,0.0,0.0,
    	4.6,	3.1,	1.5,	0.2, 1.0,0.0,0.0,
    	7.0,	3.2,	4.7,	1.4, 0.0,1.0,0.0,
    	6.9,	3.1,	4.9,	1.5, 0.0,1.0,0.0,
    	7.7,	3.8,	6.7,	2.2, 0.0,0.0,1.0,
    	7.7,	2.6,	6.9,	2.3, 0.0,0.0,1.0
    };
 */  

    /*
    for(int i = 0; i < m_e; i++)
	{	for(int j = 0; j < x_i; j++)
		cout<<entradas[i][j]<<" ";
		
		cout<<'\n';
	}*/

///////////////////////////////////////////////////// 
    /*
    double testData[][2]={
								1,      1,
                                1,      0,
                                0,      1,
                                0,      0
    };*/

    ////////////////////////////////////

    /*
    double testData[TRAINING_TEST][4] = {
    	5.1,	3.5,	1.4,	0.2,
		4.9,	3.0,	1.4,	0.2,
    	7.0,	3.2,	4.7,	1.4,
    	6.9,	3.1,	4.9,	1.5,
    	6.2,	3.4,	5.4,	2.3,
		5.9,	3.0,	5.1,	1.8

    };*/

    //int numLayers = 3, lSz[3] = {INPUT_SIZE,8,1};
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




	/*
	int count=0;
	for ( i = 0 ; i < TRAINING_TEST ; i++ ){

		bp->ffwd(testData[i]);
		double* temp=new double[10];
		for(int k=0;k<150;k++){
			temp[k]=bp->Out(k);
		}
		
		/*
		double error= error_vec(labels_vec_bin[i],temp,10);
        //if(error<=xxx){
	    if(error<=0.06){
	    	count++	;
	    }


	}
	cout<<"rate :"<< (double)count/(double)TRAINING_TEST*100<<" %"<<endl;
	*/


	
	for ( i = 0 ; i < TRAINING_TEST ; i++ )
	{
		bp->ffwd(testData[i]);
		cout<<i+1 <<" "<< testData[i][0]<< "  " << testData[i][1]<< "  " << testData[i][2]<< "  "<< testData[i][3]<< "  "<< bp->Out(0) <<" "<<bp->Out(1) << " "<<bp->Out(2)<<endl;
	}

	return 0;
}
