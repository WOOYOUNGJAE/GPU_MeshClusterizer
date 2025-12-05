#include <clusterBuilder.cuh>

#include <algorithm>
#include <chrono>

#include <vec_math.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/swap.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <unordered_set>

#include "gmcMacros.h"
#include "gmcStructs.h"
namespace gmcCuda
{
	class ScopedCPUTimer
	{
	public:
		ScopedCPUTimer()
		{
			std::string msg = name + " Starts\n";
			printf(msg.c_str());
			startTime = std::chrono::high_resolution_clock::now();
		}
		ScopedCPUTimer(const char* timerName) : name(std::string(timerName))
		{
			std::string msg = name + " Starts\n";
			printf(msg.c_str());
			startTime = std::chrono::high_resolution_clock::now();
		}
		~ScopedCPUTimer()
		{
			double duration = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - startTime).count();
			std::string msg = name + " Result : " + std::to_string(duration) + "(ms)\n";
			printf(msg.c_str());
		}
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> startTime{};
		std::string name = "Scoped CPU Timer";
	};

	namespace Kernels {

		__device__ __host__ inline uint32_t mortonExpandBits(uint32_t v) {
			v = (v * 0x00010001u) & 0xFF0000FFu;
			v = (v * 0x00000101u) & 0x0F00F00Fu;
			v = (v * 0x00000011u) & 0xC30C30C3u;
			v = (v * 0x00000005u) & 0x49249249u;
			return v;
		}

		__device__ __host__ inline uint32_t getMorton(uint3 cell) {
			const uint32_t xx = mortonExpandBits(cell.x);
			const uint32_t yy = mortonExpandBits(cell.y);
			const uint32_t zz = mortonExpandBits(cell.z);
			return xx * 4 + yy * 2 + zz; // (xx << 2) + (yy << 1) + (zz << 0)
		}

		struct MergeAABBFunctor
		{
			__host__ __device__ __forceinline__
				AABB operator()(const AABB& lhs, const AABB& rhs) const
			{
				auto b = lhs;
				b.absorb(rhs);
				return b;
			}
		};

		__global__ void Fill_AABBs(uint32_t numTriangles, const float3* vertices, const uint32_t* indexBuffer, AABB* outAABBs)
		{
			const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

			if (triangleIndex < numTriangles)
			{
				uint32_t triangle[3] = { indexBuffer[triangleIndex * 3 + 0],  indexBuffer[triangleIndex * 3 + 1] , indexBuffer[triangleIndex * 3 + 2] };
				float3 v0 = vertices[triangle[0]];
				float3 v1 = vertices[triangle[1]];
				float3 v2 = vertices[triangle[2]];

				float3 triMin = fminf(v0, fminf(v0, v1));
				float3 triMax = fmaxf(v0, fmaxf(v1, v2));

				outAABBs[triangleIndex] = AABB(triMin, triMax);
			}
		}

		__device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
			if (*addr <= value) return;
			int* addr_as_int = (int*)addr;
			int old = *addr_as_int, assumed;
			do {
				assumed = old;
				if (__int_as_float(assumed) <= value) break;
				old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
			} while (assumed != old);
		}

		__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
			if (*addr >= value) return;
			int* addr_as_int = (int*)addr;
			int old = *addr_as_int, assumed;
			do {
				assumed = old;
				if (__int_as_float(assumed) >= value) break;
				old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
			} while (assumed != old);
		}


		__device__ __forceinline__ AABB WarpReducedAABB(AABB aabb)
		{
#pragma unroll
			for (uint32_t offset = 16; offset > 0; offset >>= 1)
			{
				// Get offset lane's min/max
				float minX = __shfl_down_sync(0xFFFF'FFFF, aabb.min.x, offset);
				float minY = __shfl_down_sync(0xFFFF'FFFF, aabb.min.y, offset);
				float minZ = __shfl_down_sync(0xFFFF'FFFF, aabb.min.z, offset);

				float maxX = __shfl_down_sync(0xFFFF'FFFF, aabb.max.x, offset);
				float maxY = __shfl_down_sync(0xFFFF'FFFF, aabb.max.y, offset);
				float maxZ = __shfl_down_sync(0xFFFF'FFFF, aabb.max.z, offset);

				// Absorb
				aabb.min.x = fminf(aabb.min.x, minX);
				aabb.min.y = fminf(aabb.min.y, minY);
				aabb.min.z = fminf(aabb.min.z, minZ);

				aabb.max.x = fmaxf(aabb.max.x, maxX);
				aabb.max.y = fmaxf(aabb.max.y, maxY);
				aabb.max.z = fmaxf(aabb.max.z, maxZ);
			}
			return aabb;
		}
		__global__ void Compute_AABBs(uint32_t numTriangles, const float3* vertices, const uint32_t* indexBuffer, AABB* outAABBs)
		{
			__shared__ AABB s_AABB[gmcCuda::BLOCK_SIZE_MORTON / 32];
			const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;
			const int tIdx = threadIdx.x;
			const int laneID = tIdx % 32;
			const int warpID = tIdx / 32;
			AABB localAABB;
			// thread's AABB
			if (triangleIndex < numTriangles)
			{
				uint32_t triangle[3] = { indexBuffer[triangleIndex * 3 + 0],  indexBuffer[triangleIndex * 3 + 1] , indexBuffer[triangleIndex * 3 + 2] };
				float3 v0 = vertices[triangle[0]];
				float3 v1 = vertices[triangle[1]];
				float3 v2 = vertices[triangle[2]];

				float3 triMin = fminf(v0, fminf(v0, v1));
				float3 triMax = fmaxf(v0, fmaxf(v1, v2));
				//float3 triMin = MinFloat3(v0, v1, v2);
				//float3 triMax = MaxFloat3(v0, v1, v2);

				localAABB = AABB(triMin, triMax);
				outAABBs[triangleIndex] = localAABB;
			}
			else
			{
				localAABB = AABB();
			}
			__syncthreads();

			// Warp's AABB
			localAABB = WarpReducedAABB(localAABB);
			if (laneID == 0)
			{
				s_AABB[warpID] = localAABB;
			}
			__syncthreads();

			// Block's AABB
			if (warpID == 0) // Only Block's First Warp Do
			{
				AABB blockAABB;

				// this thread represents warp (less than num warps)
				if (tIdx < blockDim.x / 32)
				{
					blockAABB = s_AABB[tIdx]; // fetch warpAABB
				}
				blockAABB = WarpReducedAABB(blockAABB);

				// Update Global AABB
				if (tIdx == 0)
				{
					atomicMinFloat(&outAABBs[numTriangles].min.x, blockAABB.min.x);
					atomicMinFloat(&outAABBs[numTriangles].min.y, blockAABB.min.y);
					atomicMinFloat(&outAABBs[numTriangles].min.z, blockAABB.min.z);

					atomicMaxFloat(&outAABBs[numTriangles].max.x, blockAABB.max.x);
					atomicMaxFloat(&outAABBs[numTriangles].max.y, blockAABB.max.y);
					atomicMaxFloat(&outAABBs[numTriangles].max.z, blockAABB.max.z);					
				}
			}
		}

		__global__ void mortonKernel(AABB* aabbs, uint32_t* codes, uint32_t* ids, AABB rootAABB, int size)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= size) return;
			// normlize center with AABB's min,max. And * 1024
			float3 coord = rootAABB.normCoord(aabbs[tid].center()) * 1024.f;
			uint3 uiCoord = make_uint3(coord);
			codes[tid] = getMorton(clamp(uiCoord, make_uint3(0), make_uint3(1023)));
			ids[tid] = tid;
		}

		/**
		 * aabbs[numAABBs] is root AABB
		 */
		__global__ void mortonKernel(AABB* aabbs, uint32_t* codes, uint32_t* ids, int numAABBs)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numAABBs) return;
			// normlize center with AABB's min,max. And * 1024
			float3 coord = aabbs[numAABBs].normCoord(aabbs[tid].center()) * 1024.f;
			uint3 uiCoord = make_uint3(coord);
			codes[tid] = getMorton(clamp(uiCoord, make_uint3(0), make_uint3(1023)));
			ids[tid] = tid;
		}

		__global__ void Fill_ClusteredIndexBuffer_SimpleMorton(const uint32_t* __restrict__  triIDs, const uint32_t* __restrict__ oldIndexBuffer, uint32_t* __restrict__ newIndexBuffer, int numTriangles)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numTriangles) return;

			uint32_t triID = triIDs[tid];
			uint3 triangle = make_uint3(oldIndexBuffer[triID * 3 + 0], oldIndexBuffer[triID * 3 + 1], oldIndexBuffer[triID * 3 + 2]);
			newIndexBuffer[tid * 3 + 0] = triangle.x;
			newIndexBuffer[tid * 3 + 1] = triangle.y;
			newIndexBuffer[tid * 3 + 2] = triangle.z;
		}


	}

	void ClusterBuilder::Impl_MortonBased::Init_CpuPointer(const float* positions, uint32_t numPositions, uint32_t* pIndices, uint32_t numIndices)
	{
		assert(numIndices % 3 == 0);
		m_numTriangles = (numIndices / 3);

		m_pMappedIndices = pIndices;

		// Device Alloc
		CUDA_CHECK(cudaMalloc(&m_dPositions, sizeof(float3) * numPositions));
		CUDA_CHECK(cudaMalloc(&m_dOldIndexBuffer, sizeof(uint3) * m_numTriangles));

		CUDA_CHECK(cudaMalloc(&m_dNewIndexBuffer, sizeof(uint3) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dAABBs, sizeof(AABB*) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dMortons, sizeof(uint32_t*) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dTriIDs, sizeof(uint32_t*) * m_numTriangles));

		// Assign, Memcpy
		CUDA_CHECK(cudaMemcpy(m_dPositions, (float3*)positions, sizeof(float3) * numPositions, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_dOldIndexBuffer, pIndices, sizeof(uint3) * m_numTriangles, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemset(m_dNewIndexBuffer, 0xFFFF'FFFF, sizeof(uint3) * m_numTriangles));

		// Warming up
		{
			uint32_t blockSize = 256;
			dim3 gridDim = dim3(ROUND_UP_DIM(m_numTriangles, blockSize), 1, 1);
			gmcCuda::Kernels::Fill_AABBs << <gridDim, blockSize >> > (
				m_numTriangles,
				m_dPositions,
				m_dOldIndexBuffer,
				m_dAABBs);
			CUDA_SYNC_CHECK();
			thrust::reduce(
				thrust::device,
				m_dAABBs, m_dAABBs + m_numTriangles, m_rootAABB,
				Kernels::MergeAABBFunctor());
			CUDA_SYNC_CHECK();
			Kernels::mortonKernel << <gridDim, blockSize >> > (
				m_dAABBs, m_dMortons, m_dTriIDs, m_rootAABB, m_numTriangles);
			CUDA_SYNC_CHECK();
		}
	}

	void ClusterBuilder::Impl_MortonBased::Init_GpuPointer(float* mappedPositions, uint32_t numPositions,
		uint32_t* mappedIndices, uint32_t numIndices)
	{
		assert(numIndices % 3 == 0);
		m_numTriangles = (numIndices / 3);

		// Device Alloc
		CUDA_CHECK(cudaMalloc(&m_dNewIndexBuffer, sizeof(uint3) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dAABBs, sizeof(AABB) * (m_numTriangles + 1))); // last is Root AABB
		CUDA_CHECK(cudaMalloc(&m_dMortons, sizeof(uint32_t*) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dTriIDs, sizeof(uint32_t*) * m_numTriangles));

		// Map, Assign, Memcpy
		m_dPositions = (float3*)mappedPositions;
		m_dOldIndexBuffer = mappedIndices;

		CUDA_CHECK(cudaMemset(m_dNewIndexBuffer,  0xFFFF'FFFF, sizeof(uint3) * m_numTriangles));

		// Warming up
		{
			uint32_t blockSize = 256;
			dim3 gridDim = dim3(ROUND_UP_DIM(m_numTriangles, blockSize), 1, 1);
			gmcCuda::Kernels::Fill_AABBs << <gridDim, blockSize >> > (
				m_numTriangles,
				m_dPositions,
				m_dOldIndexBuffer,
				m_dAABBs);
			CUDA_SYNC_CHECK();
			thrust::reduce(
				thrust::device,
				m_dAABBs, m_dAABBs + m_numTriangles, m_rootAABB,
				Kernels::MergeAABBFunctor());
			CUDA_SYNC_CHECK();
			Kernels::mortonKernel << <gridDim, blockSize >> > (
				m_dAABBs, m_dMortons, m_dTriIDs, m_rootAABB, m_numTriangles);
			CUDA_SYNC_CHECK();
		}
	}

	uint32_t ClusterBuilder::Impl_MortonBased::BuildClusters(uint16_t clusterMaxSize, gmc::Cluster* outClusters)
	{
		//gmcCuda::GPUTimer gpuTimer(5, 0);
		gmcCuda::GPUTimer gpuTimer(4, 0);
		cudaError_t err;

		// Fill AABB
		uint32_t blockSize = 256;
		dim3 gridDim = dim3(ROUND_UP_DIM(m_numTriangles, blockSize), 1, 1);

#if (1) // Improved Version
		gpuTimer.RecordStart();
		gmcCuda::Kernels::Compute_AABBs << <gridDim, blockSize >> > (
			m_numTriangles,
			m_dPositions,
			m_dOldIndexBuffer,
			m_dAABBs);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

		// Compute morton codes.
		gpuTimer.RecordStart();
		Kernels::mortonKernel << <gridDim, blockSize >> > (
		m_dAABBs, m_dMortons, m_dTriIDs, m_numTriangles);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

#else // Old version with thrust reduction
		gpuTimer.RecordStart();
		gmcCuda::Kernels::Fill_AABBs << <gridDim, blockSize >> > (
			m_numTriangles,
			m_dPositions,
			m_dOldIndexBuffer,
			m_dAABBs);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

		// Root AABB
		m_rootAABB = AABB();
		gpuTimer.RecordStart();
		m_rootAABB = thrust::reduce(
			thrust::device,
			m_dAABBs, m_dAABBs + m_numTriangles, m_rootAABB,
			Kernels::MergeAABBFunctor());
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

		// Compute morton codes.
		gpuTimer.RecordStart();
		Kernels::mortonKernel << <gridDim, blockSize >> > (
			m_dAABBs, m_dMortons, m_dTriIDs, m_rootAABB, m_numTriangles);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();
#endif	

		// Sort morton codes
		gpuTimer.RecordStart();
		thrust::stable_sort_by_key(thrust::device, m_dMortons, m_dMortons + m_numTriangles, m_dTriIDs);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

		gpuTimer.RecordStart();
		Kernels::Fill_ClusteredIndexBuffer_SimpleMorton<<<gridDim, blockSize >>> (m_dTriIDs, m_dOldIndexBuffer, m_dNewIndexBuffer, m_numTriangles);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

		gpuTimer.printResults();

		if (!m_bClustersCreated && cudaGetLastError() == cudaSuccess)
		{
			m_bClustersCreated = true;
			// make numClusters multiple of clusterMaxSize
			m_numClusters = (m_numTriangles + clusterMaxSize - 1) / clusterMaxSize;

			for (uint32_t i = 0; i < m_numClusters - 1; ++i)
			{
				outClusters[i] = gmc::Cluster
				{
					0, clusterMaxSize * i, clusterMaxSize * 3u, clusterMaxSize
				};
			}
			// last cluster's triangle count might be less than clusterMaxSize
			uint16_t remainTriangles = m_numTriangles & (clusterMaxSize - 1);
			if (remainTriangles == 0)
			{
				m_numClusters -= 1;
			}
			else
			{
				outClusters[m_numClusters - 1] = gmc::Cluster
				{
					0, clusterMaxSize * (m_numClusters - 1), clusterMaxSize * 3u, remainTriangles
				};				
			}

			// If CPU version
			if (m_pMappedIndices)
			{
				cudaMemcpy(m_pMappedIndices, m_dNewIndexBuffer, sizeof(uint32_t) * m_numTriangles * 3, cudaMemcpyDeviceToHost);
			}

			std::swap(m_dOldIndexBuffer, m_dNewIndexBuffer);

		}

		return m_numClusters;
	}
}
