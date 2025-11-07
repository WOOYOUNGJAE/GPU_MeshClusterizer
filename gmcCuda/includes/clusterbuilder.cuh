#pragma once
#include <gmcCudaIncludes.cuh>
#include <gmcCuda/gmcCuda.h>

namespace gmcCuda
{
	class Geometry
	{
	public:
		Geometry(float* positions, uint32_t numPositions, uint32_t* indices, uint32_t numIndices)
		{
			assert(numIndices % 3 == 0);

			float3* f3Positions = (float3*)positions;
			m_dPositions.assign(f3Positions, f3Positions + numPositions);

			uint3* ui3Triangles = (uint3*)indices;
			m_dTriangles.assign(ui3Triangles, ui3Triangles + (numIndices / 3));

			m_hPositionsViewer = thrust::host_vector<float3>(m_dPositions);
			m_hTrianglesViewer = thrust::host_vector<uint3>(m_dTriangles);


		}
		~Geometry()
		{
			m_dPositions.clear();
			m_dTriangles.clear();
			m_hPositionsViewer.clear();
			m_hTrianglesViewer.clear();
		}

	public:
		thrust::device_vector<float3> m_dPositions;
		thrust::device_vector<uint3> m_dTriangles;
		

		thrust::host_vector<float3> m_hPositionsViewer;
		thrust::host_vector<uint3> m_hTrianglesViewer;

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

	public:
		Geometry* pGeometry = nullptr;
	};
}
