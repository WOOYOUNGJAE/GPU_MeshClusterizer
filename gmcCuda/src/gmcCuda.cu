#include <gmcCuda/gmcCuda.h>
#include <clusterBuilder.cuh>
#include <random>

gmcCuda::ClusterBuilder::ClusterBuilder()
{
	pImpl = new ClusterBuilder::Impl();
}

gmcCuda::ClusterBuilder::~ClusterBuilder()
{
	delete pImpl; pImpl = nullptr;
}

void gmcCuda::ClusterBuilder::Init_WithDeviceAllocation(float* positions, uint32_t numPositions, uint32_t* indices, uint32_t numIndices)
{
	pImpl->Init(positions, numPositions, indices, numIndices);
}

void gmcCuda::ClusterBuilder::BuildClusters()
{
	pImpl->BuildClusters();
}

