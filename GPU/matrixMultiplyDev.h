#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef MATRIX_DIM
#define MATRIX_DIM
// Command-line matrix dimensions (Rows, Cols, Size)
typedef struct _dim{
    unsigned long int R, C, S, mem_size;
} mdim;
#endif

int matrixMultiply(double h_A[], mdim Adim, double h_B[], mdim Bdim, double h_C[], mdim Cdim, const double alpha, const double beta, int &devID, cudaDeviceProp &deviceProp, char t);
int matrixMultiplyDev(double d_A[], mdim Adim, double d_B[], mdim Bdim, double d_C[], mdim Cdim, const double alpha, const double beta, int &devID, cudaDeviceProp &deviceProp);
int matrixMultiplyDev(double d_A[], mdim Adim, double d_B[], mdim Bdim, double d_C[], mdim Cdim, const double alpha, const double beta, int &devID, cudaDeviceProp &deviceProp, char t);
#endif
