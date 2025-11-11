#pragma once
#include <gmcCudaIncludes.cuh>
#include <gmcCuda/gmcCuda.h>

#include <lbvh.cuh>
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

	class ClusterBuilder::Impl
	{
	public:
		~Impl()
		{
			delete pGeometry; pGeometry = nullptr;
		}
		void Init(float* positions, uint32_t numPositions, uint32_t* indices, uint32_t numIndices)
		{
			pGeometry = new Geometry(positions, numPositions, indices, numIndices);
		}
		void BuildClusters()
		{
			uint32_t numTriangles = pGeometry->m_numTriangles;
			thrust::device_vector<uint2> d_res(100 * numTriangles);
			CUDA_SYNC_CHECK();

			// Build BVH
			LBVH bvh;
			printf("Building LBVH...\n");
			bvh.compute(thrust::raw_pointer_cast(pGeometry->m_dAABBs.data()), numTriangles);
			CUDA_SYNC_CHECK();

			// Query BVH
			printf("Querying LBVH...\n");
			int numCols = bvh.query(thrust::raw_pointer_cast(d_res.data()), d_res.size());
			CUDA_SYNC_CHECK();

			// Print results
			printf("Getting results...\n");
			thrust::host_vector<uint2> res(d_res.begin(), d_res.begin() + numCols);

			printf("%d collision pairs found on GPU.\n", res.size());
			// printf("GPU:\n");
			// for (size_t i = 0; i < res.size(); i++)
			// 	printf("%d %d\n", res[i].x, res[i].y);

			// Brute force compute the same result
			std::unordered_set<uint2> resSet;
			bool good = true;

			for (size_t i = 0; i < res.size(); i++) {
				uint2 a = res[i];
				if (a.x > a.y) std::swap(a.x, a.y);
				if (!resSet.insert(a).second) {
					printf("Error: Duplicate result\n");
					good = false;
				}
			}

			int numCPUFound = 0;
			printf("\nRunning brute force CPU collision detection...\n");
		}
	public:


	public:
		Geometry* pGeometry = nullptr;
	};
}
