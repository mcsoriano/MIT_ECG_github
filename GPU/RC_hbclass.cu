//v5 RESERVOIR COMPUTING IMPLEMENTATION FOR HEARTBEAT TRAINING AND CLASSIFICATION IN A GPU

#include "RC_hbclass.h"


// Main
int main(int argc, char** argv)
{
    // CPU Timer
    clock_t time_start = clock(), time_diff;

    // Time stamp
    timestamp(0);

    // Program
    printf("GPU RC heartbeat classifier program \n");

    // Initialize CUDA
    int devID = 0;
    cudaDeviceProp deviceProp;
    initializeCUDA(devID, deviceProp);

    // Declare data variables and sizes to allocate
    std::string location; // Datafile path and name

    // Alloc and Set Data
    mdim Fdim, Mdim, MBdim, Rdim, Tagdim, TrainYdim, TestYdim, TrainRdim, TestRdim;
    double *F = NULL, *M = NULL, *MB = NULL, *R = NULL, *Tag = NULL, *TrainY = NULL, *TestY = NULL, *TrainR = NULL, *TestR = NULL;
    alloc_set_data(&F, Fdim, &M, Mdim, &MB, MBdim, &R, Rdim, &Tag, Tagdim, &TrainY, TrainYdim, &TestY, TestYdim, &TrainR, TrainRdim, &TestR, TestRdim);


    //======================= FORWARD DECLARATIONS (to avoid in loop) =========
    //Timers
    //cudaEvent_t fstart, fstop;
    //float msec_cuda;

    //Input
        double* d_Rin;

        double *d_F, *d_M;
        checkCudaErrors(cudaMalloc((void**)&d_F, Fdim.mem_size));
        checkCudaErrors(cudaMalloc((void**)&d_M, Mdim.mem_size));
        checkCudaErrors(cudaMalloc((void**)&d_Rin, Rdim.mem_size));

        printf("HtD F\n");
        checkCudaErrors(cudaMemcpy(d_F, F, Fdim.mem_size, cudaMemcpyHostToDevice));

        double* d_R = NULL;
        checkCudaErrors(cudaMalloc((void**)&d_R, Rdim.mem_size));

       double *d_TrainR = NULL, *d_TestR = NULL;
       checkCudaErrors(cudaMalloc((void**)&d_TrainR, TrainRdim.mem_size));
       checkCudaErrors(cudaMalloc((void**)&d_TestR, TestRdim.mem_size));


    //INV
    mdim MtMdim;
    MtMdim.R = TrainRdim.C;
    MtMdim.C = TrainRdim.C;
    MtMdim.S = MtMdim.R * MtMdim.C;
    MtMdim.mem_size = sizeof(double) * MtMdim.S;

    double* h_INV = (double*)malloc(MtMdim.mem_size);
       double* d_MtM = NULL;
       double* d_INV = NULL;
       checkCudaErrors(cudaMalloc((void**)&d_MtM, MtMdim.mem_size));
       checkCudaErrors(cudaMalloc((void**)&d_INV, MtMdim.mem_size));


   //Weights
    mdim Weight_dim;
    Weight_dim.R = TrainRdim.C;
    Weight_dim.C = 1;
    Weight_dim.S = Weight_dim.R * Weight_dim.C;
    Weight_dim.mem_size = sizeof(double) * Weight_dim.S;
    double* h_Weight = (double*)malloc(Weight_dim.mem_size);

       double* d_TrainY = NULL;
       double* d_Waux = NULL;

       checkCudaErrors(cudaMalloc((void**)&d_TrainY, TrainYdim.mem_size));
       checkCudaErrors(cudaMalloc((void**)&d_Waux, Weight_dim.mem_size));

       printf("HtD TrainY\n");
       checkCudaErrors(cudaMemcpy(d_TrainY, TrainY, TrainYdim.mem_size, cudaMemcpyHostToDevice));


    //Classification
    double* h_TestOUT = (double*)malloc(TestYdim.mem_size);
    double* h_TrainOUT = (double*)malloc(TrainYdim.mem_size);
       double* d_TestOUT = NULL;
       double* d_TrainOUT = NULL;
       checkCudaErrors(cudaMalloc((void**)&d_TestOUT, TestYdim.mem_size));
       checkCudaErrors(cudaMalloc((void**)&d_TrainOUT, TrainYdim.mem_size));


    double exponent = exponent_val;
    double bias = bias_val;

    // No sqrt gamma
    //double gamma = gamma_val * 1.0/sqrt(Fdim.C);
    double gamma = gamma_val * 1.0;
    double eta = eta_val;


    // Assign Train/Test Submatrix rows (extended with ones)
    {
        printf("Assign Train and Test matrices\n");

        unsigned long int train_i = 0;
        unsigned long int test_i = 0;
        unsigned long int R_i = 0;
        cudaError_t err = cudaSuccess;
        double h_val = 1;
        for (long unsigned int i = 0; i < Rdim.R; i++) {
            if (Tag[i] == 0) {
                TrainR[train_i] = 1;
                err = cudaMemcpy(&d_TrainR[train_i], &h_val, sizeof(double),cudaMemcpyHostToDevice);
                train_i += TrainRdim.C;
            }
            else {
                TestR[test_i] = 1;
                err = cudaMemcpy(&d_TestR[test_i], &h_val, sizeof(double), cudaMemcpyHostToDevice);
                test_i += TestRdim.C;
            }
            R_i += Rdim.C;
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed Matrix Extension (%s)!\n",
                    cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }
    }

    printf("TrainR(%lu, %lu), TestR(%lu, %lu) \n", TrainRdim.R, TrainRdim.C, TestRdim.R, TestRdim.C);
  
    //================================= ENSEMBLE LOOP =========================
    //=========================================================================
    for (int k_iteration=1; k_iteration <= k_max_iter; k_iteration++){
    char str[50];

    //Mask
    sprintf(str, "mask%d.txt", k_iteration);
    location=Mpath;
    location += str;
    printf ("Call read routine for mask(%lu, %lu -> %lu) \n", Mdim.R, Mdim.C, Mdim.S);
    printf ("path: %s \n", location.c_str());
    read_array(M, Mdim.S, location);

    //Mask bias
    sprintf(str, "mask%d_bias.txt", k_iteration);
    location=MBpath;
    location += str;
    printf ("Call read routine for maskbias(%lu, %lu -> %lu) \n", MBdim.R, MBdim.C, MBdim.S);
    printf ("path: %s \n", location.c_str());
    read_array(MB, MBdim.S, location);

    // Reservoir
    for (unsigned long int i = 0; i < Rdim.S; i++) {
        unsigned long int j = (i % MBdim.S);
        R[i] = MB[j];
    }

    //==================================== DEVICE INPUT PRODUCT ===============
    // Reservoir Input Mask product W*M
    {
        printf("Mask product\n");
        const double alpha = 1.0;
        const double beta = 1.0;

        // copy from host
        printf("HtD B\n");
        checkCudaErrors(cudaMemcpy(d_M, M, Mdim.mem_size, cudaMemcpyHostToDevice));
        if (beta != 0) {
            printf("HtD C\n");
            checkCudaErrors(cudaMemcpy(d_Rin, R, Rdim.mem_size, cudaMemcpyHostToDevice));
        }

        matrixMultiplyDev(d_F, Fdim, d_M, Mdim, d_Rin, Rdim, alpha, beta, devID, deviceProp);
    }


    //==================================== DEVICE NONLINEAR MAPPING ===========
    // Nonlinear mapping
    printf("Nonlinear mapping\n");

    // Perform the nonlinear mapping
    nlf(d_Rin, d_TestR, TestRdim.R, TestRdim.C, exponent, bias, gamma, eta, 0);
    nlf(d_Rin, d_TrainR, TrainRdim.R, TrainRdim.C, exponent, bias, gamma, eta, TestRdim.R*Rdim.C);

    //==================================== DEVICE MtM PRODUCT =================
    // Matrix MtM 
    printf("MtM product\n");

    matrixMultiplyDev(d_TrainR, TrainRdim, d_TrainR, TrainRdim, d_MtM, MtMdim, 1.0, 0.0, devID, deviceProp, 'l');

    //==================================== DEVICE GET WEIGHTS =================
    printf("Inv*RTE*T solution\n");

    matrixMultiplyDev(d_TrainR, TrainRdim, d_TrainY, TrainYdim, d_Waux, Weight_dim, 1.0, 0.0, devID, deviceProp, 'l');

    linearSolverLU_MtM(MtMdim.R, d_MtM, Weight_dim.R, d_Waux);

//    checkCudaErrors(cudaMemcpy(h_Weight, d_Waux, Weight_dim.mem_size, cudaMemcpyDeviceToHost));


    //==================================== DEVICE CLASSIFICATION ==============
    printf("Classification\n");

    // Test
    matrixMultiplyDev(d_TestR, TestRdim, d_Waux, Weight_dim, d_TestOUT, TestYdim, 1.0, 0.0, devID, deviceProp, 'n');

    // Train
    matrixMultiplyDev(d_TrainR, TrainRdim, d_Waux, Weight_dim, d_TrainOUT, TrainYdim, 1.0, 0.0, devID, deviceProp, 'n');


    printf("DtH Class_Test\n");
    printf("DtH Class_Train\n");
    checkCudaErrors(cudaMemcpy(h_TestOUT, d_TestOUT, TestYdim.mem_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_TrainOUT, d_TrainOUT, TrainYdim.mem_size, cudaMemcpyDeviceToHost));


    // Save results

/*    sprintf(str, "weights%d.txt", k_iteration);
    location = "./";
    location += str;
    printf("Store W %s\n", location.c_str());

    Write1D(location, h_Weight, Weight_dim.S);
*/
    sprintf(str, "./class_test_%d.txt", k_iteration);
    location = str;
    Write1D(location, h_TestOUT, TestYdim.S);

/*
    sprintf(str, "./class_train_%d.txt", k_iteration);
    location = str;
    Write1D(location, h_TrainOUT, TrainYdim.S);
*/

    }
    //======================================================= ensemble_iter_end
    //=========================================================================

    //cuda Free

        // Free Multiplication arrays
        checkCudaErrors(cudaFree(d_F));
        checkCudaErrors(cudaFree(d_M));

        checkCudaErrors(cudaFree(d_Rin));

        checkCudaErrors(cudaFree(d_Waux));
        checkCudaErrors(cudaFree(d_TrainY));
        checkCudaErrors(cudaFree(d_INV));
        checkCudaErrors(cudaFree(d_MtM));

        checkCudaErrors(cudaFree(d_R));
        checkCudaErrors(cudaFree(d_TestR));
        checkCudaErrors(cudaFree(d_TrainR));
        checkCudaErrors(cudaFree(d_TestOUT));
        checkCudaErrors(cudaFree(d_TrainOUT));


    // Reset Device
    cudaDeviceReset();

    // Deallocate memory
    free(F);
    free(M);
    free(MB);
    free(R);
    free(TrainR);
    free(TestR);
    free(Tag);
    free(TrainY);
    free(TestY);
    free(h_INV);
    free(h_Weight);
    free(h_TestOUT);
    free(h_TrainOUT);


    // Stop timers
    time_diff = clock() - time_start;
    int msec = time_diff * 1000 / CLOCKS_PER_SEC;
    printf("CPU Time:  %1.3f seconds \n", msec * 1. / 1000);

    // Time stamp
    timestamp(1);

    return 0;
}



void initializeCUDA(int& devID, cudaDeviceProp& deviceProp)
{

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    error = cudaSetDevice(devID);

    if (error != cudaSuccess) {
        printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess) {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d (Warp=%d)\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.warpSize);

    return;
}


//Time stamps    //run_info.txt
void timestamp(int icall)
{
    if (icall == 0) {
        int sys0 = system("hostname > run_info.txt");
        int sys1 = system("pwd >> run_info.txt");
        int sys2 = system("date >> run_info.txt && date +%s > timestamp1");
    }
    else if (icall == 1) {
        int sys3 = system("date >> run_info.txt && date +%s > timestamp2");
        int sys4 = system("paste ./timestamp* | awk '{print $2 -$1}'>>run_info.txt");
        int sys5 = system("rm ./timestamp*");
    }
    return;
}

