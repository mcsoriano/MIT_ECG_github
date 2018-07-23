#include <cuda_runtime.h> 		//CUDA runtime routines (prefixed with "cuda_")
#include <cublas_v2.h>			//CUBLAS
#include <helper_functions.h>		//Cuda helper functions
#include <helper_cuda.h>		//Cuda helper functions
#include <stdio.h>			//Input/Output
#include <string.h>			//String objects
#include <time.h>			//Timer
#include <math.h>			//Math
#include "config.h"			//config
//#include "data_io.h"			//arrays I/O
//#include "nlf_cuda.h"
//#include "matrixMultiplyDev.h"
//#include "cuSolverDn_invMtM.h"

//Defines
#ifndef MATRIX_DIM
#define MATRIX_DIM
// Command-line matrix dimensions (Rows, Cols, Size)
typedef struct _dim{
    unsigned long int R, C, S, mem_size;
} mdim;
#endif

//Functions in code
void initializeCUDA(int &devID, cudaDeviceProp &deviceProp);
void timestamp(int icall);

//Functions in files (data_io.h, matrixMultiply, linearSolverLU)
int matrixMultiplyDev(double d_A[], mdim Adim, double d_B[], mdim Bdim, double d_C[], mdim Cdim, const double alpha, const double beta, int &devID, cudaDeviceProp &deviceProp);
int matrixMultiplyDev(double d_A[], mdim Adim, double d_B[], mdim Bdim, double d_C[], mdim Cdim, const double alpha, const double beta, int &devID, cudaDeviceProp &deviceProp, char t);
int matrixMultiply(double h_A[], mdim Adim, double h_B[], mdim Bdim, double h_C[], mdim Cdim, const double alpha, const double beta, int &devID, cudaDeviceProp &deviceProp, char t);
int linearSolverLU_MtM(int n, double *Acopy, int lda, double *b);
int nlf(double d_A[], double d_C[], int nrows, int ncols, double exponent, double bias, double gamma, double eta, int start_R);
__global__ void
vectorNLF(const double *A, const double *B, double *C, int numOps, int A_offset, int C_offset, const double *d_param);
__global__ void
updateB(double *B, const double *C, int numOps, int C_offset);

void read_array(double V[], int arr_length, std::string location);
void read_array(double V[], int arr_length, std::string location, double defval);
void Write(std::string fname, double * vect, int rows, int cols);
void Write1D(std::string fname, double * vect, int elem);
void Write1D(std::string fname, double * vect, int elem, double threshold);
void mapminmax(double V[], unsigned long int dim);
void rr_patient_average(double RR[], mdim RRdim, double Patient[], mdim Patient_dim);

//Alloc-Set Data
void alloc_set_data(double **F,	        mdim &Fdim, 
	            double **M,	        mdim &Mdim, 
                    double **MB,	mdim &MBdim, 
                    double **R,	        mdim &Rdim, 
                    double **Tag,	mdim &Tagdim, 
                    double **TrainY,	mdim &TrainYdim, 
                    double **TestY,	mdim &TestYdim, 
                    double **TrainR,	mdim &TrainRdim, 
                    double **TestR, 	mdim &TestRdim);

//Alloc Data
void alloc_data(double **A,	mdim &Adim,
              double **RR,	mdim &RRdim, 
              double **W,	mdim &Wdim, 
              double **F,	mdim &Fdim, 
	      double **M,	mdim &Mdim, 
              double **MB,	mdim &MBdim, 
              double **R,	mdim &Rdim, 
              double **Tag,	mdim &Tagdim, 
              double **Y,	mdim &Ydim, 
              double **TrainY,	mdim &TrainYdim, 
              double **TestY,	mdim &TestYdim, 
              double **TrainR,	mdim &TrainRdim, 
              double **TestR, 	mdim &TestRdim);

//Set Data
void set_data(double *A,	mdim &Adim,
              double *RR,	mdim &RRdim, 
              double *W,	mdim &Wdim, 
              double *F,	mdim &Fdim, 
	      double *M,	mdim &Mdim, 
              double *MB,	mdim &MBdim, 
              double *R,	mdim &Rdim, 
              double *Tag,	mdim &Tagdim, 
              double *Y,	mdim &Ydim, 
              double *TrainY,	mdim &TrainYdim, 
              double *TestY,	mdim &TestYdim, 
              double *TrainR,	mdim &TrainRdim, 
              double *TestR, 	mdim &TestRdim);      
