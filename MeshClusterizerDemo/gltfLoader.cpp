/**
 * glTF loading code with tinygltf is largely based on Sascha's implementation. (https://github.com/SaschaWillems/Vulkan)
 * 
 */

// Define TINYGLTF Options Before Include tinygltf.h
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE

#include "gltfLoader.h"


void GltfLoader::LoadFromFile(std::string filename)
{
	tinygltf::Model tinygltfModel;
	tinygltf::TinyGLTF gltfContext;

	size_t pos = filename.find_last_of('/');

	std::string error, warning;

	bool fileLoaded = gltfContext.LoadASCIIFromFile(&tinygltfModel, &error, &warning, filename);
	if (!fileLoaded)
	{
		std::cerr << "gltf load failed!\n";
		return;
	}
	const tinygltf::Scene& scene = tinygltfModel.scenes[tinygltfModel.defaultScene > -1 ? tinygltfModel.defaultScene : 0];
	for (size_t i = 0; i < scene.nodes.size(); ++i)
	{
		const tinygltf::Node tinygltfNode = tinygltfModel.nodes[scene.nodes[i]];
		LoadNode(nullptr, tinygltfNode, scene.nodes[i], tinygltfModel, m_indices, m_vertices);
	}
	uint32_t numVertices = m_vertices.size();
	m_vPositions.resize(numVertices);
	for (uint32_t i = 0; i < numVertices; ++i)
	{
		memcpy(&m_vPositions[i], &m_vertices[i].pos, sizeof(glm::vec3));
	}
}

void GltfLoader::LoadNode(Node* pParent, const tinygltf::Node& tinygltfNode, uint32_t nodeIndex,
	const tinygltf::Model& tinygltfModel, std::vector<uint32_t>& indices, std::vector<VertexSimple>& vertices)
{
	Node* newNode = new Node();
	newNode->pParent = pParent;
	newNode->pMesh = nullptr;
	newNode->index = nodeIndex;
	newNode->matrix = glm::mat4(1.f);

	uint32_t numMeshVertices = 0;
	// Generate local node matrix
	glm::vec3 translation = glm::vec3(0.0f);
	glm::mat4 rotation = glm::mat4(1.0f);
	glm::vec3 scale = glm::vec3(1.0f);
	if (tinygltfNode.translation.size() == 3)
	{
		translation = glm::make_vec3(tinygltfNode.translation.data());
	}
	if (tinygltfNode.rotation.size() == 4)
	{
		glm::quat q = glm::make_quat(tinygltfNode.rotation.data());
	}
	if (tinygltfNode.scale.size() == 3) 
	{
		scale = glm::make_vec3(tinygltfNode.scale.data());
	}
	if (tinygltfNode.matrix.size() == 16) 
	{
		newNode->matrix = glm::make_mat4x4(tinygltfNode.matrix.data());
	}

	// Node with children
	if (tinygltfNode.children.size() > 0) 
	{
		for (size_t i = 0; i < tinygltfNode.children.size(); i++) 
		{
			LoadNode(newNode, tinygltfModel.nodes[tinygltfNode.children[i]], tinygltfNode.children[i], tinygltfModel, indices, vertices);
		}
	}

	// Node contains mesh data
	if (tinygltfNode.mesh > -1) {
		static uint32_t meshID = 0;
		const tinygltf::Mesh mesh = tinygltfModel.meshes[tinygltfNode.mesh];
		bool hasSkin = false;
		Mesh* newMesh = new Mesh();
		for (size_t j = 0; j < mesh.primitives.size(); j++) {
			const tinygltf::Primitive& primitive = mesh.primitives[j];
			if (primitive.indices < 0) {
				continue;
			}
			uint32_t indexStart = static_cast<uint32_t>(indices.size());
			uint32_t vertexStart = static_cast<uint32_t>(vertices.size());
			uint32_t indexCount = 0;
			uint32_t vertexCount = 0;
			glm::vec3 posMin{};
			glm::vec3 posMax{};
			// Vertices
			{
				const float* bufferPos = nullptr;
				const float* bufferNormals = nullptr;
				const float* bufferTexCoords = nullptr;
				const float* bufferColors = nullptr;
				const float* bufferTangents = nullptr;
				uint32_t numColorComponents;
				const uint16_t* bufferJoints = nullptr;
				const float* bufferWeights = nullptr;

				// Position attribute is required
				assert(primitive.attributes.find("POSITION") != primitive.attributes.end());

				const tinygltf::Accessor& posAccessor = tinygltfModel.accessors[primitive.attributes.find("POSITION")->second];
				const tinygltf::BufferView& posView = tinygltfModel.bufferViews[posAccessor.bufferView];
				bufferPos = reinterpret_cast<const float*>(&(tinygltfModel.buffers[posView.buffer].data[posAccessor.byteOffset + posView.byteOffset]));
				posMin = glm::vec3(posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);
				posMax = glm::vec3(posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);

				if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
					const tinygltf::Accessor& normAccessor = tinygltfModel.accessors[primitive.attributes.find("NORMAL")->second];
					const tinygltf::BufferView& normView = tinygltfModel.bufferViews[normAccessor.bufferView];
					bufferNormals = reinterpret_cast<const float*>(&(tinygltfModel.buffers[normView.buffer].data[normAccessor.byteOffset + normView.byteOffset]));
				}

				if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
					const tinygltf::Accessor& uvAccessor = tinygltfModel.accessors[primitive.attributes.find("TEXCOORD_0")->second];
					const tinygltf::BufferView& uvView = tinygltfModel.bufferViews[uvAccessor.bufferView];
					bufferTexCoords = reinterpret_cast<const float*>(&(tinygltfModel.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
				}

				if (primitive.attributes.find("COLOR_0") != primitive.attributes.end())
				{
					const tinygltf::Accessor& colorAccessor = tinygltfModel.accessors[primitive.attributes.find("COLOR_0")->second];
					const tinygltf::BufferView& colorView = tinygltfModel.bufferViews[colorAccessor.bufferView];
					// Color buffer are either of type vec3 or vec4
					numColorComponents = colorAccessor.type == TINYGLTF_PARAMETER_TYPE_FLOAT_VEC3 ? 3 : 4;
					bufferColors = reinterpret_cast<const float*>(&(tinygltfModel.buffers[colorView.buffer].data[colorAccessor.byteOffset + colorView.byteOffset]));
				}

				if (primitive.attributes.find("TANGENT") != primitive.attributes.end())
				{
					const tinygltf::Accessor& tangentAccessor = tinygltfModel.accessors[primitive.attributes.find("TANGENT")->second];
					const tinygltf::BufferView& tangentView = tinygltfModel.bufferViews[tangentAccessor.bufferView];
					bufferTangents = reinterpret_cast<const float*>(&(tinygltfModel.buffers[tangentView.buffer].data[tangentAccessor.byteOffset + tangentView.byteOffset]));
				}

				// Skinning
				// Joints
				if (primitive.attributes.find("JOINTS_0") != primitive.attributes.end()) {
					const tinygltf::Accessor& jointAccessor = tinygltfModel.accessors[primitive.attributes.find("JOINTS_0")->second];
					const tinygltf::BufferView& jointView = tinygltfModel.bufferViews[jointAccessor.bufferView];
					bufferJoints = reinterpret_cast<const uint16_t*>(&(tinygltfModel.buffers[jointView.buffer].data[jointAccessor.byteOffset + jointView.byteOffset]));
				}

				if (primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end()) {
					const tinygltf::Accessor& uvAccessor = tinygltfModel.accessors[primitive.attributes.find("WEIGHTS_0")->second];
					const tinygltf::BufferView& uvView = tinygltfModel.bufferViews[uvAccessor.bufferView];
					bufferWeights = reinterpret_cast<const float*>(&(tinygltfModel.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
				}

				hasSkin |= (bufferJoints && bufferWeights);

				vertexCount = static_cast<uint32_t>(posAccessor.count);

				for (size_t v = 0; v < posAccessor.count; v++) {
					/*
					 * if skin:VertexSkiniing, else: VertexSimple
					 * allocated in here, released in "loadfromfile()"
					 * pushed into param::vertexBuffer
					 */
					VertexSimple vert{};

					vert.pos = glm::vec4(glm::make_vec3(&bufferPos[v * 3]), 1.0f);
					//if (bool preTransform) // apply node's transform to vertices while loading
					//{
					//	vert->pos = newNode->getMatrix() * glm::vec4(vert->pos, 1.f);
					//}

					vert.normal = glm::normalize(glm::vec3(bufferNormals ? glm::make_vec3(&bufferNormals[v * 3]) : glm::vec3(0.0f)));
					vert.uv = bufferTexCoords ? glm::make_vec2(&bufferTexCoords[v * 2]) : glm::vec3(0.0f);
					if (bufferColors) {
						switch (numColorComponents) {
						case 3:
							vert.color = glm::vec4(glm::make_vec3(&bufferColors[v * 3]), 1.0f);
							break;
						case 4:
							vert.color = glm::make_vec4(&bufferColors[v * 4]);
							break;
						}
					}
					else {
						vert.color = glm::vec4(1.0f);
					}
					//vert->tangent = bufferTangents ? glm::vec4(glm::make_vec4(&bufferTangents[v * 4])) : glm::vec4(0.0f);
					if (hasSkin)
					{
						//static_cast<VertexSkinning*>(vert)->joint0 = glm::vec4(glm::make_vec4(&bufferJoints[v * 4]));
						uint8_t* ptr = (uint8_t*)bufferJoints;
						(vert).joint0 = glm::vec4(glm::make_vec4(&ptr[v * 4]));
						(vert).weight0 = glm::vec4(glm::make_vec4(&bufferWeights[v * 4]));
#if CUSTOM_VERTEX
						(vert).customData4.x = meshID;
#endif
					}
					vertices.push_back(vert);
				}
			}
			// Indices
			{
				const tinygltf::Accessor& accessor = tinygltfModel.accessors[primitive.indices];
				const tinygltf::BufferView& bufferView = tinygltfModel.bufferViews[accessor.bufferView];
				const tinygltf::Buffer& buffer = tinygltfModel.buffers[bufferView.buffer];

				indexCount = static_cast<uint32_t>(accessor.count);

				switch (accessor.componentType) {
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
					uint32_t* buf = new uint32_t[accessor.count];
					memcpy(buf, &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(uint32_t));
					for (size_t index = 0; index < accessor.count; index++) {
						indices.push_back(buf[index] + vertexStart);
					}
					delete[] buf;
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
					uint16_t* buf = new uint16_t[accessor.count];
					memcpy(buf, &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(uint16_t));
					for (size_t index = 0; index < accessor.count; index++) {
						indices.push_back(buf[index] + vertexStart);
					}
					delete[] buf;
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
					uint8_t* buf = new uint8_t[accessor.count];
					memcpy(buf, &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(uint8_t));
					for (size_t index = 0; index < accessor.count; index++) {
						indices.push_back(buf[index] + vertexStart);
					}
					delete[] buf;
					break;
				}
				default:
					std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
					return;
				}
			}
			GltfPrimitive* newPrimitive = new GltfPrimitive{};
			newPrimitive->firstVertex = vertexStart;
			newPrimitive->vertexCount = vertexCount;
			newMesh->primitives.push_back(newPrimitive);
			numMeshVertices += vertexCount;
		}
		++meshID;
		newMesh->numVertices = numMeshVertices;
		newNode->pMesh = newMesh;
		m_linearMeshes.push_back(newMesh);
	}
	if (pParent) {
		pParent->children.push_back(newNode);
	}
	else {
		m_NodeTree.push_back(newNode);
	}
	m_lineaerNodes.push_back(newNode);
}
