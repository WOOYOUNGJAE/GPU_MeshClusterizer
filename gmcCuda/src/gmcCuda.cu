#include <gmcCuda/gmcCuda.h>
#include <clusterbuilder.cuh>
#include <lbvh.cuh>
#include <random>

gmcCuda::ClusterBuilder::ClusterBuilder()
{
	pImple = new ClusterBuilder::Impl();
}

gmcCuda::ClusterBuilder::~ClusterBuilder()
{
	delete pImple; pImple = nullptr;
}

void gmcCuda::ClusterBuilder::Init_WithDeviceAllocation(float* positions, uint32_t numPositions, uint32_t* indices, uint32_t numIndices)
{
	pImple->Init(positions, numPositions, indices, numIndices);
	//pImple->pGeometry.
	//gmcCuda::Geometry
}

void gmcCuda::ClusterBuilder::BuildClusters()
{
	// calc triangle's centroid
	//pImple->pGeometry
}

