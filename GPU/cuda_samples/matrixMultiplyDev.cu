/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//Source code adapted from CUDA Samples

//Include header (helper functions and CUBLAS)
#include "../matrixMultiplyDev.h"
int matrixMultiplyDev(double d_A[], mdim Adim, double d_B[], mdim Bdim, double d_C[], mdim Cdim, const double alpha, const double beta, int &devID, cudaDeviceProp &deviceProp)
{
    if (&deviceProp==NULL) printf("CudaProp ERROR \n");

    printf("MatrixA(%lu,%lu), MatrixB(%lu,%lu), MatrixC(%lu,%lu)\n",
           Adim.R, Adim.C,
           Bdim.R, Bdim.C,
           Cdim.R, Cdim.C);
    printf("SizeA(%lu), SizeB(%lu), SizeC(%lu)\n", Adim.S, Bdim.S, Cdim.S);

    
    if( Adim.C != Bdim.R ||
        Adim.R != Cdim.R ||
        Bdim.C != Cdim.C)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }

    // create and start timer
    printf("Computing result using CUBLAS...");

    // execute the kernel
    int nIter = 1;

    // CUBLAS version 2.0
    {
        //const double alpha = 1.0;
        //const double beta  = 1.0;
        cublasHandle_t handle;
        cudaEvent_t start, stop;

        checkCudaErrors(cublasCreate(&handle));

        // Allocate CUDA event timers
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        // Record the start event
        checkCudaErrors(cudaEventRecord(start, NULL));

	//Call multiplication
        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Bdim.C, Adim.R, Adim.C, &alpha, d_B, Bdim.C, d_A, Adim.C, &beta, d_C, Bdim.C));
        }

        printf("done.\n");

        // Record the stop event
        checkCudaErrors(cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * Cdim.R * Cdim.C * Bdim.R;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9) / (msecPerMatrixMul / 1000.0);
        printf(
            "Performance= %.2f GFlop/s (double), Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));
    }

    bool resCUBLAS = true;

    // cudaDeviceReset causes the driver to clean up all state
    // profile data to be flushed before the application exits
//    cudaDeviceReset();

    if (resCUBLAS == true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }
}
 

///////////////////////////////////////////////////////////////////////
int matrixMultiplyDev(double d_A[], mdim Adim, double d_B[], mdim Bdim, double d_C[], mdim Cdim, const double alpha, const double beta, int &devID, cudaDeviceProp &deviceProp, char t)
{
if (&deviceProp==NULL) printf("CudaProp ERROR \n");

unsigned long int lda=Adim.C;
unsigned long int ldb=Bdim.C; 
unsigned long int ldc=Cdim.C;

if (t=='l'){
unsigned long int aux = Adim.R;
    Adim.R=Adim.C;
    Adim.C=aux;
    printf("%c case, swap A\n", t);
}

else if (t=='r'){
unsigned long int aux = Bdim.R;
    Bdim.R=Bdim.C;
    Bdim.C=aux;
    printf("%c case, swap B\n", t);
}

else if (t=='a'){
unsigned long int aux;
    aux = Adim.R;
    Adim.R=Adim.C;
    Adim.C=aux;
    aux = Bdim.R;
    Bdim.R=Bdim.C;
    Bdim.C=aux;
    printf("%c case, swap A and B\n", t);
}

    printf("MatrixA(%lu,%lu), MatrixB(%lu,%lu), MatrixC(%lu,%lu)\n",
           Adim.R, Adim.C,
           Bdim.R, Bdim.C,
           Cdim.R, Cdim.C);
    printf("SizeA(%lu), SizeB(%lu), SizeC(%lu)\n", Adim.S, Bdim.S, Cdim.S);

    
    if( Adim.C != Bdim.R ||
        Adim.R != Cdim.R ||
        Bdim.C != Cdim.C)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
 
   printf("lda(%lu), ldb(%lu), ldc(%lu)\n", lda, ldb, ldc);

    // create and start timer
    printf("Computing result using CUBLAS...");

    // execute the kernel
    int nIter = 1;

    // CUBLAS version 2.0
    {
        //const double alpha = 1.0;
        //const double beta  = 1.0;
        cublasHandle_t handle;
        cudaEvent_t start, stop;

        checkCudaErrors(cublasCreate(&handle));

        // Allocate CUDA event timers
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        // Record the start event
        checkCudaErrors(cudaEventRecord(start, NULL));

	//Call multiplication
        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            if (t=='n') checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Bdim.C, Adim.R, Adim.C, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc));
            else if (t=='l') checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Bdim.C, Adim.R, Adim.C, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc));
            else if (t=='r') checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Bdim.C, Adim.R, Adim.C, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc));
            else if (t=='a') checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, Bdim.C, Adim.R, Adim.C, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc));
	    else return EXIT_FAILURE;
        }
        printf("done.\n");

        // Record the stop event
        checkCudaErrors(cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * Cdim.R * Cdim.C * Bdim.R;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9) / (msecPerMatrixMul / 1000.0);
        printf(
            "Performance= %.2f GFlop/s (double), Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));
    }

    bool resCUBLAS = true;

    // cudaDeviceReset causes the driver to clean up all state
    // profile data to be flushed before the application exits
//    cudaDeviceReset();

    if (resCUBLAS == true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }
}
 


///////////////////////////////////////////////////////////////////////
int matrixMultiply(double h_A[], mdim Adim, double h_B[], mdim Bdim, double h_C[], mdim Cdim, const double alpha, const double beta, int &devID, cudaDeviceProp &deviceProp, char t)
{
    if (&deviceProp==NULL) printf("CudaProp ERROR \n");

unsigned long int lda=Adim.C;
unsigned long int ldb=Bdim.C; 
unsigned long int ldc=Cdim.C;

if (t=='l'){
unsigned long int aux = Adim.R;
    Adim.R=Adim.C;
    Adim.C=aux;
    printf("%c case, swap A\n", t);
}

else if (t=='r'){
unsigned long int aux = Bdim.R;
    Bdim.R=Bdim.C;
    Bdim.C=aux;
    printf("%c case, swap B\n", t);
}

else if (t=='a'){
unsigned long int aux;
    aux = Adim.R;
    Adim.R=Adim.C;
    Adim.C=aux;
    aux = Bdim.R;
    Bdim.R=Bdim.C;
    Bdim.C=aux;
    printf("%c case, swap A and B\n", t);
}

    printf("MatrixA(%lu,%lu), MatrixB(%lu,%lu), MatrixC(%lu,%lu)\n",
           Adim.R, Adim.C,
           Bdim.R, Bdim.C,
           Cdim.R, Cdim.C);
    printf("SizeA(%lu), SizeB(%lu), SizeC(%lu)\n", Adim.S, Bdim.S, Cdim.S);

    
    if( Adim.C != Bdim.R ||
        Adim.R != Cdim.R ||
        Bdim.C != Cdim.C)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
 
   printf("lda(%lu), ldb(%lu), ldc(%lu)\n", lda, ldb, ldc);
    unsigned long int mem_size_A = sizeof(double) * Adim.S;
    unsigned long int mem_size_B = sizeof(double) * Bdim.S;
    unsigned long int mem_size_C = sizeof(double) * Cdim.S;

    // allocate device memory and copy from host
    double *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    printf("HtD A\n");
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    printf("HtD B\n");
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    if (beta!=0){
    printf("HtD C\n");
    checkCudaErrors(cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice));
    }

    // create and start timer
    printf("Computing result using CUBLAS...");

    // execute the kernel
    int nIter = 1;

    // CUBLAS version 2.0
    {
        //const double alpha = 1.0;
        //const double beta  = 1.0;
        cublasHandle_t handle;
        cudaEvent_t start, stop;

        checkCudaErrors(cublasCreate(&handle));

        // Allocate CUDA event timers
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        // Record the start event
        checkCudaErrors(cudaEventRecord(start, NULL));

	//Call multiplication
        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            if (t=='n') checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Bdim.C, Adim.R, Adim.C, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc));
            else if (t=='l') checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Bdim.C, Adim.R, Adim.C, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc));
            else if (t=='r') checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Bdim.C, Adim.R, Adim.C, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc));
            else if (t=='a') checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, Bdim.C, Adim.R, Adim.C, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc));
	    else return EXIT_FAILURE;
        }

        printf("done.\n");

        // Record the stop event
        checkCudaErrors(cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * Cdim.R * Cdim.C * Bdim.R;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9) / (msecPerMatrixMul / 1000.0);
        printf(
            "Performance= %.2f GFlop/s (double), Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

        // copy result from device to host
        printf("DtH C\n");
        checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));
    }

    bool resCUBLAS = true;
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    // cudaDeviceReset causes the driver to clean up all state
    // profile data to be flushed before the application exits
    // cudaDeviceReset();

    if (resCUBLAS == true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }
}
 
