#pragma once
#include <vector>
#include <string>
#include <iostream>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#include "tiny_gltf.h"
/**
 * Simple gltf Loader, only loading vertices and indices
 */
class GltfLoader
{
public: // TypeDefs
	typedef struct VERTEX_TYPE
	{
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 uv;
		glm::vec4 color;
		glm::vec4 joint0;
		glm::vec4 weight0;
#if CUSTOM_VERTEX
		glm::vec4 customData4; // [meshID, primitiveIdInMesh, 0, 0]
#endif
	}VertexType, VertexSimple;
private:
	struct GltfPrimitive
	{
		uint32_t firstIndex;
		uint32_t indexCount;
		uint32_t firstVertex;
		uint32_t vertexCount;
	};
	struct Mesh
	{
		std::vector<GltfPrimitive*> primitives;
		uint32_t numVertices = 0;
	};
	struct Node
	{
		Node* pParent;
		Mesh* pMesh;
		uint32_t index;
		std::vector<Node*> children;
		glm::mat4 matrix;
		std::string name;
	};
public:
	GltfLoader() = default;
	~GltfLoader(){}

public: // Funcs
	void LoadFromFile(std::string filename);
	void LoadNode(Node* pParent, const tinygltf::Node& tinygltfNode, uint32_t nodeIndex, const tinygltf::Model& tinygltfModel, std::vector<uint32_t>& indices, std::vector<VertexSimple>& vertices);

public: // Members
	std::vector<VertexSimple> m_vertices;
	std::vector<glm::vec3> m_vPositions;
	std::vector<uint32_t> m_indices;
	std::vector<const Mesh*> m_linearMeshes; // read Only
	std::vector<Node*> m_NodeTree;
	std::vector<Node*> m_lineaerNodes;
	//std::vector<GltfPrimitive> m_indices;


};

