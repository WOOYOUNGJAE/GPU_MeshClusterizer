#pragma once
#include <gmcCudaIncludes.cuh>
#include <gmcCuda/gmcCuda.h>

#include <unordered_set>

#include "gmcStructs.h"


namespace gmcCuda
{
	constexpr uint32_t BLOCK_SIZE_MORTON = 256;

	/// <summary>
	/// Very simple high-performance GPU LBVH that takes in a list of bounding boxes and outputs overlapping pairs.
	/// Side note: null bounds (inf, -inf) as inputs are ignored automatically.
	/// Base of class LBVH is from Jerry Hsu, 2024(https://github.com/jerry060599/KittenGpuLBVH/blob/main/lbvh.cuh)
	/// </summary>
	class LBVH {
	public:
		typedef AABB aabb;

		// 64 byte node struct. Can fit two in a 128 byte cache line.
		struct alignas(64) node {
			uint32_t parentIdx;			// Parent node. Most siginificant bit (MSB) is used to indicate whether this is a left or right child of said parent.
			uint32_t leftIdx;			// Index of left child node. MSB is used to indicate whether this is a leaf node.
			uint32_t rightIdx;			// Index of right child node. MSB is used to indicate whether this is a leaf node.
			uint32_t fence;				// This subtree have indices between fence and current index.

			aabb bounds[2];
		};

	private:
		struct thrustImpl;
		std::unique_ptr<thrustImpl> impl;

		size_t numObjs = 0;
		aabb rootBounds;
		// This is exactly how large a stack needs to be to traverse this tree.
		// Used by query() to minimize register usage.
		int maxStackSize = 1;

	public:
		LBVH();
		~LBVH();

		// Returns the total bounds of every node in this tree.
		aabb bounds();

		/// <summary>
		/// Refits an existing aabb tree once compute() has been called.
		/// Does not recompute the tree structure but only the AABBs.
		/// </summary>
		void refit();

		/// <summary>
		/// Allocates memory and builds the LBVH from a list of AABBs.
		/// Can be called multiple times for memory reuse.
		/// </summary>
		/// <param name="devicePtr">The device pointer containing the AABBs</param>
		/// <param name="size">The number of AABBs</param>
		void compute(aabb* devicePtr, size_t size, uint16_t clusterSize, uint32_t* d_oldIndexBuffer, uint32_t* d_newIndexBuffer);

		struct IsValidNodeFunctor
		{
			const LBVH::node* nodes;
			uint2* nodeRanges;
			__host__ __device__
				IsValidNodeFunctor(const LBVH::node* _nodes, uint2* _nodeRanges) : nodes(_nodes), nodeRanges(_nodeRanges) {}

			__host__ __device__ bool operator()(int i);
		};
	};


	class ClusterBuilder::Impl
	{
	public:
		~Impl()
		{
			cudaFree(m_dRootAABB);
			cudaFree(m_dTriIDs);
			cudaFree(m_dMortons);
			cudaFree(m_dAABBs);
			cudaFree(m_dNewIndexBuffer);
		}
		void Init(float* positions, uint32_t numPositions, uint32_t* indices, uint32_t numIndices);
		void Init_WithExternalMappedMemory(float* mappedPositions, uint32_t numPositions, uint32_t* mappedIndices, uint32_t numIndices);

		void BuildClusters();
		/**
		 * Simply cluster based on sorted Morton codes
		 * @return num of clusters
		 */
		uint32_t BuildClusters_SimpleMorton(uint16_t clusterMaxSize, gmc::Cluster* outClusters);


	public:
		LBVH* m_pLBVH = nullptr;
		bool m_bInit = false;
		uint16_t clusterSize = 64;
		uint32_t m_numTriangles = 0;

		float3* m_dPositions; // Original Vertices
		uint32_t* m_dOldIndexBuffer = nullptr;
		uint32_t* m_dNewIndexBuffer = nullptr; // Clustered Index Buffer
	public: // for clustering process
		AABB m_rootAABB;
		AABB* m_dRootAABB;
		AABB* m_dAABBs = nullptr;
		uint32_t* m_dMortons = nullptr;
		uint32_t* m_dTriIDs = nullptr;
	private:
		bool m_bClustersCreated = false;
		uint32_t m_numClusters = 0;
	private:
	};
}
