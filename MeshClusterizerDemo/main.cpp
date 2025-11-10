#include "gltfLoader.h"
#include "gmcCuda/gmcCuda.h"

int main(int argc, char* argv[])
{
	GltfLoader gltfLoader;
	gltfLoader.LoadFromFile("D:\\Documents\\Blender\\Exports\\MocapGuy.gltf");

	uint32_t numPositions = gltfLoader.m_vPositions.size();

	gmcCuda::ClusterBuilder clusterBuilder;
	clusterBuilder.Init_WithDeviceAllocation((float*)gltfLoader.m_vPositions.data(), numPositions, gltfLoader.m_indices.data(), gltfLoader.m_indices.size());
	clusterBuilder.BuildClusters();
	return 0;
}
