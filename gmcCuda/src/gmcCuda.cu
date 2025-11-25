#include <gmcCuda/gmcCuda.h>
#include <gmcStructs.h>
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

void gmcCuda::ClusterBuilder::Init_WithExternalMappedMemory(float* mappedPositions, uint32_t numPositions,
	uint32_t* mappedIndices, uint32_t numIndices)
{
	pImpl->Init_WithExternalMappedMemory(mappedPositions, numPositions, mappedIndices, numIndices);
}

void gmcCuda::ClusterBuilder::BuildClusters()
{
	pImpl->BuildClusters();
}

uint32_t gmcCuda::ClusterBuilder::BuildClusters_SimpleMorton(uint16_t clusterMaxSize, gmc::Cluster* outClusters)
{
	return pImpl->BuildClusters_SimpleMorton(clusterMaxSize, outClusters);
}

