#pragma once
//#include <Windows.h>
//static __int64 _start, _freq, _end;
//static float _compute_time;
//#define CHECK_TIME_START(start,freq) QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
//#define CHECK_TIME_END(start,end,freq,time) QueryPerformanceCounter((LARGE_INTEGER*)&end); time = (float)((float)(end - start) / (freq * 1.0e-3f))
//

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n",          \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                   \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Kernel Error: %s (error code %d) at %s:%d\n",   \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CUDA_SYNC_CHECK()                                                         \
    do {                                                                          \
        cudaDeviceSynchronize();                                                  \
        CUDA_CHECK_LAST_ERROR();                                                  \
    } while (0)