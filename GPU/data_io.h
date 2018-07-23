#ifndef DATA_IO_H
#define DATA_IO_H

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>

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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include "config.h"
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
#endif
