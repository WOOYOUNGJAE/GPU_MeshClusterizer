#pragma once

namespace gmc // Common Namespace
{
	struct Cluster
	{
		/* offsets within meshlet_vertices and meshlet_triangles arrays with meshlet data */
		unsigned int vertex_offset;
		unsigned int triangle_offset;

		/* number of vertices and triangles used in the meshlet; data is stored in consecutive range defined by offset and count */
		unsigned int vertex_count;
		unsigned int triangle_count;
	};	
}