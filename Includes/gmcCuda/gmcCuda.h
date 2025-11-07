#pragma once
/**
 * No Dependancy for Cuda, glm. Just c++
 */
#include <gmcDefines.h>
#include <cstdint>

namespace gmcCuda
{
	// build cluster with triangle
	GMC_DLL void BuildClusters(float* positions, uint32_t numPositions, uint32_t indices, uint32_t numIndices);
}