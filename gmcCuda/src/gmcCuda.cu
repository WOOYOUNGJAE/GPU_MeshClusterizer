#include <gmcCuda/gmcCuda.h>
#include <gmcStructs.h>
#include <clusterBuilder.cuh>
#include <random>

gmcCuda::ClusterBuilder::ClusterBuilder()
{
	pImpl_mortonBased = new ClusterBuilder::Impl_MortonBased();
}

gmcCuda::ClusterBuilder::~ClusterBuilder()
{
	delete pImpl_mortonBased; pImpl_mortonBased = nullptr;
}

void gmcCuda::ClusterBuilder::Init_MortonBased_CpuPointer(const float* positions, uint32_t numPositions, uint32_t* pIndices, uint32_t numIndices)
{
	pImpl_mortonBased->Init_CpuPointer(positions, numPositions, pIndices, numIndices);
}

void gmcCuda::ClusterBuilder::Init_MortonBased_GpuPointer(float* mappedPositions, uint32_t numPositions,
	uint32_t* mappedIndices, uint32_t numIndices)
{
	pImpl_mortonBased->Init_GpuPointer(mappedPositions, numPositions, mappedIndices, numIndices);
}

uint32_t gmcCuda::ClusterBuilder::BuildClusters_MortonBased(uint16_t clusterMaxSize, gmc::Cluster* outClusters)
{
	return pImpl_mortonBased->BuildClusters(clusterMaxSize, outClusters);
}

