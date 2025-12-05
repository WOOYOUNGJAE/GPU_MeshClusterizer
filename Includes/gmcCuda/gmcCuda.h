#pragma once
/**
 * No Dependancy for Cuda, glm. Just c++
 */
#include <gmcDefines.h>
#include <cstdint>

namespace gmc
{
	struct Cluster;
}

namespace gmcCuda
{
	/**
	 * Only a subset of functions and kernels are exposed to the Importer.
	 * The actual implementation is in the "Impl" class.
	 */
	class GMC_DLL ClusterBuilder
	{
	public:
		ClusterBuilder();
		~ClusterBuilder();
	public:
		/**
		 * @brief Device Allocation in Cuda, and final Written on CPU Index Array
		 * @param positions CPU float3 array
		 * @param numPositions 
		 * @param pIndices  CPU uint array
		 * @param numIndices 
		 */
		void Init_MortonBased_CpuPointer(const float* positions, uint32_t numPositions, uint32_t* pIndices, uint32_t numIndices);
		/**
		 * @brief Write on DEVICE INDEX BUFFERs (External Mapped Memory)
		 * @param mappedPositions mapped device float3 memory (vk, d3d12, ,,,)
		 * @param numPositions
		 * @param mappedIndices mapped device memory uint memory (vk, d3d12, ,,,)
		 * @param numIndices 
		 */
		void Init_MortonBased_GpuPointer(float* mappedPositions, uint32_t numPositions, uint32_t* mappedIndices, uint32_t numIndices);
		/**
		 * @param clusterMaxSize num of max triangles
		 * @param outClusters 
		 * @return num of clusters
		 */
		uint32_t BuildClusters_MortonBased(uint16_t clusterMaxSize, gmc::Cluster* outClusters);

	private:
		class Impl_MortonBased;
		Impl_MortonBased* pImpl_mortonBased = nullptr; // PIMPL
	};
}