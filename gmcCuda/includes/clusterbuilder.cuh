#pragma once
#include <gmcCudaIncludes.cuh>
#include <gmcCuda/gmcCuda.h>

#include <unordered_set>


namespace gmcCuda
{
	__global__ inline void Calculate_Mortons(uint32_t numTriangles, const float3* vertices, const uint3* triangles, float3* outCentroids)
	{
		const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

		if (triangleIndex < numTriangles)
		{
			uint3 triangle = triangles[triangleIndex];
			float3 v0 = vertices[triangle.x];
			float3 v1 = vertices[triangle.y];
			float3 v2 = vertices[triangle.z];

			float3 pos = (1.f / 3.f) * (v0 + v1 + v2);
		}
	}
	__device__ inline float3 Calculate_Morton(int triangleIndex, uint32_t numTriangles, const float3* vertices, const uint3* triangles)
	{
		uint3 triangle = triangles[triangleIndex];
		float3 v0 = vertices[triangle.x];
		float3 v1 = vertices[triangle.y];
		float3 v2 = vertices[triangle.z];

		float3 centroid = (1.f / 3.f) * (v0 + v1 + v2);
		return centroid;
	}

	__global__ inline void Fill_AABBs(uint32_t numTriangles, const float3* vertices, const uint3* triangles, AABB* outAABBs)
	{
		const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

		if (triangleIndex < numTriangles)
		{
			//float3 centroid = Calculate_Morton(triangleIndex, numTriangles, vertices, triangles);
			uint3 triangle = triangles[triangleIndex];
			float3 v0 = vertices[triangle.x];
			float3 v1 = vertices[triangle.y];
			float3 v2 = vertices[triangle.z];

			float3 triMin = MinFloat3(v0, v1, v2);
			float3 triMax = MaxFloat3(v0, v1, v2);

			outAABBs[triangleIndex] = AABB(triMin, triMax);
		}
	}

	class Geometry
	{
	public: // Launch Dimension
		static constexpr uint32_t BLOCK_SIZE = 1024;
		dim3 m_gridDim;
	public:
		Geometry(float* positions, uint32_t numPositions, uint32_t* indices, uint32_t numIndices)
		{
			assert(numIndices % 3 == 0);
			uint32_t numTriangles = (numIndices / 3);
			m_numTriangles = numTriangles;

			// Launch Dimension
			m_gridDim = dim3(ROUND_UP_DIM(numTriangles, BLOCK_SIZE), 1, 1);

			// Device Vectors
			float3* f3Positions = (float3*)positions;
			m_dPositions.assign(f3Positions, f3Positions + numPositions);

			uint3* ui3Triangles = (uint3*)indices;
			m_dTriangles.assign(ui3Triangles, ui3Triangles + numTriangles);


			// Host viewers
			m_hPositionsViewer = thrust::host_vector<float3>(m_dPositions);
			m_hTrianglesViewer = thrust::host_vector<uint3>(m_dTriangles);
			cudaDeviceSynchronize();

			m_dAABBs.resize(numTriangles);
			gmcCuda::Fill_AABBs << <m_gridDim, BLOCK_SIZE >> > (
				numTriangles,
				thrust::raw_pointer_cast(m_dPositions.data()),
				thrust::raw_pointer_cast(m_dTriangles.data()),
				thrust::raw_pointer_cast(m_dAABBs.data())
				);
			CUDA_SYNC_CHECK();
			m_hAABBsViewer = thrust::host_vector<AABB>(m_dAABBs);
		}
		~Geometry()
		{
			m_dAABBs.clear();
			m_dPositions.clear();	
			m_dTriangles.clear();
			m_hAABBsViewer.clear();
			m_hPositionsViewer.clear();
			m_hTrianglesViewer.clear();
		}

	public: // thrust
		thrust::device_vector<float3> m_dPositions; // Original Vertices
		thrust::device_vector<uint3> m_dTriangles;
		thrust::device_vector<AABB> m_dAABBs;

		thrust::host_vector<float3> m_hPositionsViewer;
		thrust::host_vector<uint3> m_hTrianglesViewer;
		thrust::host_vector<AABB> m_hAABBsViewer;
	public:
		uint32_t m_numTriangles = 0;

	};


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
			delete m_pGeometry; m_pGeometry = nullptr;
		}
		void Init(float* positions, uint32_t numPositions, uint32_t* indices, uint32_t numIndices)
		{
			m_pGeometry = new Geometry(positions, numPositions, indices, numIndices);
		}
		void BuildClusters();

	public:


	public:
		Geometry* m_pGeometry = nullptr;
		LBVH* m_pLBVH = nullptr;

		uint16_t clusterSize = 64;
		uint32_t* m_dOldIndexBuffer = nullptr;
		uint32_t* m_dNewIndexBuffer = nullptr; // Clustered Index Buffer
	};
}
