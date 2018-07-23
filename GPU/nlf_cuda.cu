//Include header (helper functions and nlf)
#include "nlf_cuda.h"
int nlf(double d_A[], double d_C[], int nrows, int ncols, double exponent, double bias, double gamma, double eta, int start_R)
{

    //Get nlf row-wise Reservoir

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    unsigned long int numElements = nrows * ncols;
    int numOps = ncols-1;
    size_t sizeB = numOps * sizeof(double);
    printf("[Vector function of %lu elements]\n", numElements);

    // Set the parameter vector
    double param[4] = {exponent, bias, gamma, eta};

    // Allocate the host input vector B
    double* h_B = (double*)malloc(sizeB);

    // Verify that allocations succeeded
    if (h_B == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numOps; ++i) {
        h_B[i] = 0;
    }

    // Allocate the device input vector B
    double* d_B = NULL;
    err = cudaMalloc((void**)&d_B, sizeB);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors (host memory to the device memory)
    printf("HtD nlf B\n");
    err = cudaMemcpy(d_B, &h_B[0], sizeB, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_param
    double* d_param = NULL;
    err = cudaMalloc((void**)&d_param, 4*sizeof(double));

    // Copy the host input vectors (host memory to the device memory)
    printf("HtD nlf param\n");
    err = cudaMemcpy(d_param, &param[0], 4*sizeof(double), cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector param from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    // Launch the Vector NLF CUDA Kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = (numOps + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    unsigned long int C_offset;
    unsigned long int A_offset;

    //ROW LOOP
    for (unsigned long int i_row = 0; i_row < nrows; i_row++) {
	C_offset = i_row * ncols + 1;
        A_offset = i_row * (ncols-1) + start_R;
        vectorNLF<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numOps, A_offset, C_offset, d_param);
        err = cudaGetLastError();

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch vectorNLF kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Launch the Update Kernel
        updateB<<<blocksPerGrid, threadsPerBlock>>>(d_B, d_C, numOps, C_offset);
        err = cudaGetLastError();

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch updateB kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    printf("Total rows processed: %d \n", nrows);

    // Free device global memory
    err = cudaFree(d_B);
    err = cudaFree(d_param);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_B);

    printf("Done\n");
    return 0;
}

// Nonlinear function kernel
// Parameters passed d_param={exponent, bias, gamma, eta}
__global__ void
vectorNLF(const double *A, const double *B, double *C, int numOps, int A_offset, int C_offset, const double *d_param)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = (i - 1 + numOps) % numOps;
    if (i < numOps) {
        C[i + C_offset] = 1.0 / (1.0 + exp(-d_param[0] * ( d_param[2] * A[i + A_offset] + d_param[3] * B[j]))) - d_param[1];
    }
}

// Nonlinear function kernel update (circular)
__global__ void
updateB(double* B, const double* C, int numOps, int C_offset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = (i - 1 + numOps) % numOps;

    if (i < numOps) {
        B[j] = C[j + C_offset];
    }
}

