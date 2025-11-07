#include "gltfLoader.h"
#include "gmcCuda/gmcCuda.h"

int main(int argc, char* argv[])
{
	GltfLoader gltfLoader;
	gltfLoader.LoadFromFile("D:\\Documents\\Blender\\Exports\\MocapGuy.gltf");

	uint32_t numPositions = gltfLoader.m_vPositions.size();


	gmcCuda::BuildClusters(gltfLoader.m_vPositions, gltfLoader.m_indices);
	//gmcCuda::BuildClusters((float*)gltfLoader.m_vPositions.data(), numPositions, (unsigned int*)gltfLoader.m_indices.data(), gltfLoader.m_indices.size());
	//lbvh::bvh<float, float4, aabb_getter> bvh(f4Positions.begin(), f4Positions.end(), true);


	return 0;
}
