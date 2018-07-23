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

//Include header (helper functions and CUSOLVE)
#include "../cuSolverDn_invMtM.h"
int linearSolverLU_MtM(int n, double *Acopy, int lda, double *b){
    cusolverDnHandle_t handle;
    checkCudaErrors(cusolverDnCreate(&handle));
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    double start, stop;
    double time_solve;
    cudaEvent_t cstart, cstop;


    // 1 - Get Buffer
    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));

    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    int nIter=1;
    start = second();
    start = second();
    checkCudaErrors(cudaEventCreate(&cstart));
    checkCudaErrors(cudaEventCreate(&cstop));
    checkCudaErrors(cudaEventRecord(cstart, NULL));

    // 2 - Factorize    
    for (int ni=0; ni<nIter; ni++){
    checkCudaErrors(cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));
    if ( 0 != h_info ){
        printf("Error: LU factorization failed\n");
    }

    // 3 - Solve
    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, b, n, info));
    checkCudaErrors(cudaDeviceSynchronize());

    }
    stop = second();
    checkCudaErrors(cudaEventRecord(cstop, NULL));
    time_solve = 1./nIter * (stop - start);
    printf ("timing: LU = %10.6f sec\n", time_solve);
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, cstart, cstop));
    printf ("timing(msec): LU = %10.6f msec\n", msecTotal*1./nIter);

    //Store LU
    //double *hh_A = NULL; 
    //hh_A = (double*)malloc(sizeof(double)*n*n);
    //checkCudaErrors(cudaMemcpy(hh_A, A, sizeof(double)*n*n, cudaMemcpyDeviceToHost));
    //
    //printf(" ----------------- \n"); 
    //std::string fname="./lu_A.txt";
    //Write(fname, hh_A, n, n);
    //
    ////Free
    //if (hh_A  ) { free(hh_A);}


    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (ipiv  ) { checkCudaErrors(cudaFree(ipiv));}

    return 0;
}


