#ifndef BACK_PROPAGATION_H
#define BACK_PROPAGATION_H

#include <cassert>
#include <iostream>
#include <cstdio>
#include <cmath>


class Back_Propagation{

	double **out;

	double **delta;

	double ***weight;

	int numl;

	int *lsize;

	double beta;

	double alpha;

	double ***prevDwt;

	double sigmoid(double in);

public:

	~Back_Propagation();

	Back_Propagation(int nl,int *sz,double b,double a);

	void bpgt(double *in,double *tgt);

	void ffwd(double *in);

	double mse(double *tgt) const;

	double Out(int i) const;
};

#endif // BACK_PROPAGATION_H
