#ifndef CUSOLVERDN_INVMTM_H
#define CUSOLVERDN_INVMTM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"

int linearSolverLU_MtM(
    int n,
    double *Acopy,
    int lda,
    double *b);

#endif
