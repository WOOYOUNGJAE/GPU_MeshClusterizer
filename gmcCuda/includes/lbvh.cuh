#pragma once
#include <gmcCudaIncludes.cuh>
// Base of class LBVH is from Jerry Hsu, 2024(https://github.com/jerry060599/KittenGpuLBVH/blob/main/lbvh.cuh)


namespace gmcCuda {

	/// <summary>
	/// Very simple high-performance GPU LBVH that takes in a list of bounding boxes and outputs overlapping pairs.
	/// Side note: null bounds (inf, -inf) as inputs are ignored automatically.
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
			//uint32_t range_fence;		// first 16bits for range, other 16bits for fence.

			aabb bounds[2];
		};

	private:
		struct thrustImpl;
		std::unique_ptr<thrustImpl> impl;

		size_t numObjs = 0;
		aabb rootBounds;
		uint16_t clusterSize = 64;
		uint32_t* d_oldIndexBuffer = nullptr;
		uint32_t* d_newIndexBuffer = nullptr; // Clustered Index Buffer
		// This is exactly how large a stack needs to be to traverse this tree.
		// Used by query() to minimize register usage.
		int maxStackSize = 1;

	public:
		LBVH(uint32_t* d_IndexBuffer);
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
		void compute(aabb* devicePtr, size_t size);

		/// <summary>
		/// Tests this BVH against another BVH. Outputs unique collision pairs.
		/// The calling BVH should be the smaller one for best performance. 
		/// </summary>
		/// <param name="d_res">Device pointer with pairs containing (in order) the calling BVH object ID and then the other BVH object ID.</param>
		/// <param name="resSize">The number of entries allocated</param>
		/// <param name="other">The other BVH</param>
		/// <returns>The number of unique collision pairs written</returns>
		size_t query(uint2* d_res, size_t resSize, LBVH* other) const;

		/// <summary>
		/// Tests this BVH against itself. Outputs unique collision pairs.
		/// </summary>
		/// <param name="d_res">Device pointer with unique object ID pairs</param>
		/// <param name="resSize">The number of entries allocated</param>
		/// <returns>The number of unique collision pairs written</returns>
		size_t query(uint2* d_res, size_t resSize) const;

		// Does a self check of the BVH structure for debugging purposes.
		void bvhSelfCheck() const;

		struct IsValidNodeFunctor
		{
			const LBVH::node* nodes;
			uint2* nodeRanges;
			__host__ __device__
				IsValidNodeFunctor(const LBVH::node* _nodes, uint2* _nodeRanges) : nodes(_nodes), nodeRanges(_nodeRanges) {}

			__host__ __device__ bool operator()(int i);
		};
	};

	// Tests the LBVH with a simple test case of 100k objects.
	void testLBVH();
}