#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#define THREADS_PER_BLOCK 512 // most last-gen GPUs support up to 1024 threads per block

#define OCCUPIED_FLAG (1 << 0)
#define FREE_FLAG (1 << 1)
#define FREE_RAY_FLAG (1 << 2)
#define UPDATED_FLAG (1 << 3)

#ifndef __CUDA_EXEC_SPACE__
    #ifdef __CUDACC__ // when compiling on NVCC
        #define __CUDA_EXEC_SPACE__ __host__ __device__ // this function will be compiled for both host and device
    #else
        #define __CUDA_EXEC_SPACE__
    #endif
#endif

#endif