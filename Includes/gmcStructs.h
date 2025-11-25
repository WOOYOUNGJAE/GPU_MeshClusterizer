#pragma once

namespace gmc // Common Namespace
{
	struct Cluster
	{
		unsigned int vertexOffset; // start offset from global vertex buffer
		unsigned int triangleOffset; // start offset from global Index buffer

		unsigned int vertexCount;
		unsigned int triangleCount;
	};	
}