#pragma once
/**
 * No Dependancy for Cuda, glm. Just c++
 */
#include <gmcDefines.h>
#include <cstdint>

namespace gmcCuda
{
	// 
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
		// Allocate Geometry Data(Vertex Positions, Indices,,) to CUDA
		void Init_WithDeviceAllocation(float* positions, uint32_t numPositions, uint32_t* indices, uint32_t numIndices);

		// build cluster with triangle
		void BuildClusters();

	private:
		class Impl;
		Impl* pImpl = nullptr; // PIMPL
	};
	
}