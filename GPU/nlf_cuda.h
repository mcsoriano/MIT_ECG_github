//Include header (helper functions and nlf)
#ifndef NLF_H
#define NLF_H

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>


int nlf(double d_A[], double d_C[], int nrows, int ncols, double exponent, double bias, double gamma, double eta, int start_R);

__global__ void
vectorNLF(const double *A, const double *B, double *C, int numOps, int A_offset, int C_offset, const double *d_param);

__global__ void
updateB(double* B, const double* C, int numOps, int C_offset);

#endif

