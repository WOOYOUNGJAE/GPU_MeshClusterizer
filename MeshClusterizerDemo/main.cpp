#include "gltfLoader.h"
#include "gmcCuda/gmcCuda.h"
#include <gmcStructs.h>

int main(int argc, char* argv[])
{
	GltfLoader gltfLoader;
	gltfLoader.LoadFromFile("D:\\Documents\\Blender\\Exports\\MocapGuy.gltf");

	uint32_t numPositions = gltfLoader.m_vPositions.size();

	gmcCuda::ClusterBuilder clusterBuilder;
	clusterBuilder.Init_WithDeviceAllocation((float*)gltfLoader.m_vPositions.data(), numPositions, gltfLoader.m_indices.data(), gltfLoader.m_indices.size());


	uint32_t clusterMaxSize = 64;
	std::vector<gmc::Cluster> clusters(gltfLoader.m_indices.size() / (clusterMaxSize * 2));
	uint32_t numClusters = clusterBuilder.BuildClusters_SimpleMorton(clusterMaxSize, clusters.data());


	return 0;
}
