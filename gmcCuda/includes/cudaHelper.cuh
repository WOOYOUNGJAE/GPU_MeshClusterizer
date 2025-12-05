#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n",          \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);            \
            /*exit(EXIT_FAILURE);*/                                                   \
        }                                                                         \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                   \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Kernel Error: %s (error code %d) at %s:%d\n",   \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);            \
            /*exit(EXIT_FAILURE);*/                                                   \
        }                                                                         \
    } while (0)

#define CUDA_SYNC_CHECK()                                                         \
    do {                                                                          \
        cudaDeviceSynchronize();                                                  \
        CUDA_CHECK_LAST_ERROR();                                                  \
    } while (0)

#define ROUND_UP_DIM(N, S) ((((N) + (S) - 1) / (S)))

#define DEBUG_WATCH_CUDA_MEM(DST, SRC, TOTAL_SIZE) \
cudaMemcpy((void*)DST, SRC, TOTAL_SIZE, cudaMemcpyDeviceToHost);

#define CUDA_RELEASE(dPtr) \
    if (dPtr) {cudaFree(dPtr); (dPtr) = nullptr;}

namespace gmcCuda
{
#define GMC_MEASURE_MODE 0
	class GPUTimer
	{
	public:
        GPUTimer(uint32_t _numQueies, cudaStream_t _targetStream = 0) : numQueries(_numQueies), targetStream(_targetStream)
        {
#if GMC_MEASURE_MODE
            startArr = new cudaEvent_t[numQueries];
            endArr = new cudaEvent_t[numQueries];
            resultArr = new float[numQueries];
	        for (uint32_t i = 0; i < numQueries; ++i)
	        {
                resultArr[i] = -FLT_MAX;
                CUDA_CHECK(cudaEventCreate(&startArr[i]));
                CUDA_CHECK(cudaEventCreate(&endArr[i]));
	        }
#endif
        }
        ~GPUTimer()
        {
#if GMC_MEASURE_MODE
            delete[] resultArr; resultArr = nullptr;
            for (uint32_t i = 0; i < numQueries; ++i)
            {
                CUDA_CHECK(cudaEventDestroy(startArr[i]));
                CUDA_CHECK(cudaEventDestroy(endArr[i]));
            }
            delete[] startArr; startArr = nullptr;
            delete[] endArr; endArr = nullptr;
#endif
        }
	public:
        void RecordStart()
        {
#if GMC_MEASURE_MODE
            CUDA_CHECK(cudaEventRecord(startArr[curQueryIndex], targetStream));
#endif
        }
        void RecordEnd()
        {
#if GMC_MEASURE_MODE
            CUDA_CHECK(cudaEventRecord(endArr[curQueryIndex++], targetStream));
#endif
        }
        void collectResults()
		{
#if GMC_MEASURE_MODE
            CUDA_CHECK(cudaStreamSynchronize(targetStream));
            for (uint32_t i = 0; i < curQueryIndex; ++i)
            {
                cudaEventElapsedTime(resultArr + i, startArr[i], endArr[i]);
            }
            curQueryIndex = 0;
#endif
		}
        void printResults()
        {
#if GMC_MEASURE_MODE
            if (curQueryIndex > 0)
                collectResults();

            for (uint32_t i = 0; i < numQueries; ++i)
            {
                printf("GPU Timer %d : %f\n", i, resultArr[i]);
            }
#endif
        }
	public:
        float* resultArr = nullptr;
	private:
        cudaStream_t targetStream;
        cudaEvent_t* startArr = nullptr;
        cudaEvent_t* endArr = nullptr;

        uint32_t curQueryIndex = 0;
        uint32_t numQueries = 0;
	};
}