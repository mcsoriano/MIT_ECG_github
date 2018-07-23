# Make sure CUDA paths are added
# export PATH=/usr/local/cuda-9.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
# https://docs.nvidia.com/cuda/eula/index.html

# Location of the CUDA Toolkit and paths
CUDA_PATH ?= /usr/local/cuda-9.0

# Flags
NVCCFLAGS := -m64 -Wno-deprecated-gpu-targets -O2
INCLUDES  := -I/usr/local/cuda-9.0/samples/common/inc
LIBRARIES := -lcusolver -lcublas -lcudart -lcusparse


# Target rules

all: RC_hbclass.x

data_io.o: data_io.cu data_io.h config.h
	nvcc $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

nlf_cuda.o: nlf_cuda.cu nlf_cuda.h
	nvcc $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

cuSolverDn_invMtM.o: ./cuda_samples/cuSolverDn_invMtM.cu cuSolverDn_invMtM.h
	nvcc $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

matrixMultiplyDev.o: ./cuda_samples/matrixMultiplyDev.cu matrixMultiplyDev.h
	nvcc $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

RC_hbclass.o: RC_hbclass.cu RC_hbclass.h config.h
	nvcc $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

RC_hbclass.x: RC_hbclass.o data_io.o nlf_cuda.o matrixMultiplyDev.o cuSolverDn_invMtM.o
	nvcc $(NVCCFLAGS) -o $@ $+ $(LIBRARIES)

run: all
	./RC_hbclass.x

clean:
	rm -f RC_hbclass.x *.o 

cleanobj:
	rm -f *.o 

rebuild: clean all

fresh: clean run

# clear: 
#	$(shell clear)
