#pragma once

#pragma nv_diag_suppress esa_on_defaulted_function_ignored
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <gmcCuda/gmcCuda.h>
#include <cudaHelper.cuh>
#include "gmcStructs.h"


namespace gmcCuda
{
	constexpr uint32_t BLOCK_SIZE_MORTON = 256;

	/**
	 * Most of this struct's codes are from KittenGpuLBVH(https://github.com/jerry060599/KittenGpuLBVH)'s Bound
	 */
	struct AABB {
		float3 min;
		float3 max;

		__host__ __device__
			AABB()
			: min(make_float3(INFINITY, INFINITY, INFINITY)),
			max(make_float3(-INFINITY, -INFINITY, -INFINITY)) {
		}

		__host__ __device__
			AABB(const float3& center)
			: min(center), max(center) {
		}

		__host__ __device__
			AABB(const float3& min_, const float3& max_)
			: min(min_), max(max_) {
		}

		__host__ __device__
			AABB(const AABB& b)
			: min(b.min), max(b.max) {
		}

		__host__ __device__ inline
			float3 center() const {
			return make_float3(
				0.5f * (min.x + max.x),
				0.5f * (min.y + max.y),
				0.5f * (min.z + max.z)
			);
		}

		__host__ __device__ inline
			void absorb(const AABB& b) {
			min.x = fminf(min.x, b.min.x); min.y = fminf(min.y, b.min.y); min.z = fminf(min.z, b.min.z);
			max.x = fmaxf(max.x, b.max.x); max.y = fmaxf(max.y, b.max.y); max.z = fmaxf(max.z, b.max.z);
		}

		__host__ __device__ inline
			void absorb(const float3& p) {
			min.x = fminf(min.x, p.x); min.y = fminf(min.y, p.y); min.z = fminf(min.z, p.z);
			max.x = fmaxf(max.x, p.x); max.y = fmaxf(max.y, p.y); max.z = fmaxf(max.z, p.z);
		}

		__host__ __device__ inline
			bool contains(const float3& p) const {
			return (min.x <= p.x && p.x <= max.x) &&
				(min.y <= p.y && p.y <= max.y) &&
				(min.z <= p.z && p.z <= max.z);
		}

		__host__ __device__ inline
			bool contains(const AABB& b) const {
			return contains(b.min) && contains(b.max);
		}

		__host__ __device__ inline
			bool intersects(const AABB& b) const {
			return !(max.x < b.min.x || min.x > b.max.x ||
				max.y < b.min.y || min.y > b.max.y ||
				max.z < b.min.z || min.z > b.max.z);
		}

		__host__ __device__ inline
			void pad(float padding) {
			min.x -= padding; min.y -= padding; min.z -= padding;
			max.x += padding; max.y += padding; max.z += padding;
		}

		__host__ __device__ inline
			void pad(const float3& padding) {
			min.x -= padding.x; min.y -= padding.y; min.z -= padding.z;
			max.x += padding.x; max.y += padding.y; max.z += padding.z;
		}

		__host__ __device__ inline
			float volume() const {
			float dx = max.x - min.x;
			float dy = max.y - min.y;
			float dz = max.z - min.z;
			return dx * dy * dz;
		}

		__host__ __device__ inline
			float3 normCoord(const float3& pos) const {
			return make_float3(
				(pos.x - min.x) / (max.x - min.x),
				(pos.y - min.y) / (max.y - min.y),
				(pos.z - min.z) / (max.z - min.z)
			);
		}

		__host__ __device__ inline
			float3 interp(const float3& coord) const {
			return make_float3(
				min.x + (max.x - min.x) * coord.x,
				min.y + (max.y - min.y) * coord.y,
				min.z + (max.z - min.z) * coord.z
			);
		}
	};

	class ClusterBuilder::Impl_MortonBased
	{
	public:
		Impl_MortonBased() = default;
		~Impl_MortonBased()
		{
			// If true, this is CPU Version
			// Else : gpu ptr. release from graphics api
			if (m_pMappedIndices)
			{
				CUDA_RELEASE(m_dOldIndexBuffer);
				CUDA_RELEASE(m_dPositions);
			}
			CUDA_RELEASE(m_dRootAABB);
			CUDA_RELEASE(m_dTriIDs);
			CUDA_RELEASE(m_dMortons);
			CUDA_RELEASE(m_dAABBs);
			CUDA_RELEASE(m_dNewIndexBuffer);
		}
	public:
		void Init_CpuPointer(const float* positions, uint32_t numPositions, uint32_t* pIndices, uint32_t numIndices);
		void Init_GpuPointer(float* mappedPositions, uint32_t numPositions, uint32_t* mappedIndices, uint32_t numIndices);
		/**
		 * Simply cluster based on sorted Morton codes
		 * @return num of clusters
		 */
		uint32_t BuildClusters(uint16_t clusterMaxSize, gmc::Cluster* outClusters);


	private:
		uint32_t m_numClusters = 0;
		uint32_t m_numTriangles = 0;
		bool m_bClustersCreated = false;
		AABB m_rootAABB;
	private: // cpu version only
		uint32_t* m_pMappedIndices = nullptr;
	private: // device ptr
		float3* m_dPositions; // Original Vertices
		uint32_t* m_dOldIndexBuffer = nullptr;
		uint32_t* m_dNewIndexBuffer = nullptr; // Clustered Index Buffer
		AABB* m_dRootAABB;
		AABB* m_dAABBs = nullptr;
		uint32_t* m_dMortons = nullptr;
		uint32_t* m_dTriIDs = nullptr;
	};


}
