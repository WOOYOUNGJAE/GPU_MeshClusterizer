#include <clusterBuilder.cuh>

#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/swap.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cub/cub.cuh>

#include <unordered_set>

#include "gmcMacros.h"
#include "gmcStructs.h"

namespace gmcCuda
{
	class ScopedCPUTimer
	{
	public:
		ScopedCPUTimer()
		{
			std::string msg = name + " Starts\n";
			printf(msg.c_str());
			startTime = std::chrono::high_resolution_clock::now();
		}
		ScopedCPUTimer(const char* timerName) : name(std::string(timerName))
		{
			std::string msg = name + " Starts\n";
			printf(msg.c_str());
			startTime = std::chrono::high_resolution_clock::now();
		}
		~ScopedCPUTimer()
		{
			double duration = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - startTime).count();
			std::string msg = name + " Result : " + std::to_string(duration) + "(ms)\n";
			printf(msg.c_str());
		}
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> startTime{};
		std::string name = "Scoped CPU Timer";
	};

	struct LBVH::thrustImpl {
		thrust::device_ptr<aabb> d_objs = nullptr;
		thrust::device_vector<int> d_flags;					// Flags used for updating the tree

		thrust::device_vector<uint32_t> d_morton;			// Morton codes for each object
		thrust::device_vector<uint32_t> d_objIDs;			// Object ID for each leaf
		thrust::device_vector<uint32_t> d_leafParents;		// Parent ID for each leaf. MSB is used to indicate whether this is a left or right child of said parent.
		thrust::device_vector<node> d_nodes;				// The internal tree nodes

		// node's covering ranges for leaves. y>=x is guaranteed
		// uint2.x's msb : whether this node is a candidate for clustering.
		thrust::device_vector<uint2> d_nodeRanges;
		thrust::device_vector<uint32_t> d_clusterTriCounter;
	};


	namespace Kernels {
		struct MergeAABBFunctor
		{
			__host__ __device__ __forceinline__
				AABB operator()(const AABB& lhs, const AABB& rhs) const
			{
				auto b = lhs;
				b.absorb(rhs);
				return b;
			}
		};

		__device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
			if (*addr <= value) return;
			int* addr_as_int = (int*)addr;
			int old = *addr_as_int, assumed;
			do {
				assumed = old;
				if (__int_as_float(assumed) <= value) break;
				old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
			} while (assumed != old);
		}

		__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
			if (*addr >= value) return;
			int* addr_as_int = (int*)addr;
			int old = *addr_as_int, assumed;
			do {
				assumed = old;
				if (__int_as_float(assumed) >= value) break;
				old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
			} while (assumed != old);
		}


		__device__ __forceinline__ AABB WarpReducedAABB(AABB aabb)
		{
#pragma unroll
			for (uint32_t offset = 16; offset > 0; offset >>= 1)
			{
				// Get offset lane's min/max
				float minX = __shfl_down_sync(0xFFFF'FFFF, aabb.min.x, offset);
				float minY = __shfl_down_sync(0xFFFF'FFFF, aabb.min.y, offset);
				float minZ = __shfl_down_sync(0xFFFF'FFFF, aabb.min.z, offset);

				float maxX = __shfl_down_sync(0xFFFF'FFFF, aabb.max.x, offset);
				float maxY = __shfl_down_sync(0xFFFF'FFFF, aabb.max.y, offset);
				float maxZ = __shfl_down_sync(0xFFFF'FFFF, aabb.max.z, offset);

				// Absorb
				aabb.min.x = fminf(aabb.min.x, minX);
				aabb.min.y = fminf(aabb.min.y, minY);
				aabb.min.z = fminf(aabb.min.z, minZ);

				aabb.max.x = fmaxf(aabb.max.x, maxX);
				aabb.max.y = fmaxf(aabb.max.y, maxY);
				aabb.max.z = fmaxf(aabb.max.z, maxZ);
			}
			return aabb;
		}
		__global__ void Fill_AABBs(uint32_t numTriangles, const float3* vertices, const uint32_t* indexBuffer, AABB* outAABBs)
		{
			const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

			if (triangleIndex < numTriangles)
			{
				uint32_t triangle[3] = { indexBuffer[triangleIndex * 3 + 0],  indexBuffer[triangleIndex * 3 + 1] , indexBuffer[triangleIndex * 3 + 2] };
				float3 v0 = vertices[triangle[0]];
				float3 v1 = vertices[triangle[1]];
				float3 v2 = vertices[triangle[2]];

				float3 triMin = MinFloat3(v0, v1, v2);
				float3 triMax = MaxFloat3(v0, v1, v2);

				outAABBs[triangleIndex] = AABB(triMin, triMax);
			}
		}

		__global__ void Compute_AABBs(uint32_t numTriangles, const float3* vertices, const uint32_t* indexBuffer, AABB* outAABBs)
		{
			__shared__ AABB s_AABB[gmcCuda::BLOCK_SIZE_MORTON / 32];
			const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;
			const int tIdx = threadIdx.x;
			const int laneID = tIdx % 32;
			const int warpID = tIdx / 32;
			AABB localAABB;
			// thread's AABB
			if (triangleIndex < numTriangles)
			{
				uint32_t triangle[3] = { indexBuffer[triangleIndex * 3 + 0],  indexBuffer[triangleIndex * 3 + 1] , indexBuffer[triangleIndex * 3 + 2] };
				float3 v0 = vertices[triangle[0]];
				float3 v1 = vertices[triangle[1]];
				float3 v2 = vertices[triangle[2]];

				float3 triMin = MinFloat3(v0, v1, v2);
				float3 triMax = MaxFloat3(v0, v1, v2);

				localAABB = AABB(triMin, triMax);
				outAABBs[triangleIndex] = localAABB;
			}
			else
			{
				localAABB = AABB();
			}
			__syncthreads();

			// Warp's AABB
			localAABB = WarpReducedAABB(localAABB);
			if (laneID == 0)
			{
				s_AABB[warpID] = localAABB;
			}
			__syncthreads();

			// Block's AABB
			if (warpID == 0) // Only Block's First Warp Do
			{
				AABB blockAABB;

				// this thread represents warp (less than num warps)
				if (tIdx < blockDim.x / 32)
				{
					blockAABB = s_AABB[tIdx]; // fetch warpAABB
				}
				blockAABB = WarpReducedAABB(blockAABB);

				// Update Global AABB
				if (tIdx == 0)
				{
					atomicMinFloat(&outAABBs[numTriangles].min.x, blockAABB.min.x);
					atomicMinFloat(&outAABBs[numTriangles].min.y, blockAABB.min.y);
					atomicMinFloat(&outAABBs[numTriangles].min.z, blockAABB.min.z);

					atomicMaxFloat(&outAABBs[numTriangles].max.x, blockAABB.max.x);
					atomicMaxFloat(&outAABBs[numTriangles].max.y, blockAABB.max.y);
					atomicMaxFloat(&outAABBs[numTriangles].max.z, blockAABB.max.z);					
				}
			}
		}


		__global__ void Update_Centroids(uint32_t numTriangles, const float3* vertices, const uint3* triangles, float* outCentroids)
		{
			const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

			if (triangleIndex < numTriangles)
			{
				//float3 centroid = Calculate_Morton(triangleIndex, numTriangles, vertices, triangles);
				uint3 triangle = triangles[triangleIndex];
				float3 v0 = vertices[triangle.x];
				float3 v1 = vertices[triangle.y];
				float3 v2 = vertices[triangle.z];

				float3 triMin = MinFloat3(v0, v1, v2);
				float3 triMax = MaxFloat3(v0, v1, v2);

				//outCentroids[triangleIndex] = AABB(triMin, triMax);
			}
		}

		__global__ void mortonKernel(const float3* vertices, const uint3* triangles, const float3* centroids, uint32_t* codes, uint32_t* ids, uint32_t numTriangles, AABB wholeAABB)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numTriangles) return;
			// normlize center with AABB's min,max. And * 1024
			float3 coord = wholeAABB.normCoord(centroids[tid]) * 1024.f;
			uint3 uiCoord = make_uint3(coord);
			codes[tid] = getMorton(clamp(uiCoord, make_uint3(0), make_uint3(1023)));
			ids[tid] = tid;
		}
		
		__global__ void mortonKernel(AABB* aabbs, uint32_t* codes, uint32_t* ids, AABB wholeAABB, int size)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= size) return;
			// normlize center with AABB's min,max. And * 1024
			float3 coord = wholeAABB.normCoord(aabbs[tid].center()) * 1024.f;
			uint3 uiCoord = make_uint3(coord);
			codes[tid] = getMorton(clamp(uiCoord, make_uint3(0), make_uint3(1023)));
			ids[tid] = tid;
		}

		__global__ void mortonKernel(AABB* aabbs, uint32_t* codes, uint32_t* ids, int size)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= size) return;
			// normlize center with AABB's min,max. And * 1024
			float3 coord = aabbs[size].normCoord(aabbs[tid].center()) * 1024.f;
			uint3 uiCoord = make_uint3(coord);
			codes[tid] = getMorton(clamp(uiCoord, make_uint3(0), make_uint3(1023)));
			ids[tid] = tid;
		}

		// Uses CUDA intrinsics for counting leading zeros
		__device__ inline int commonUpperBits(const uint64_t lhs, const uint64_t rhs) {
			return ::__clzll(lhs ^ rhs);
		}

		// Merges morton code with its index to output a sorted unique 64-bit key.
		__device__ inline uint64_t mergeIdx(const uint32_t code, const int idx) {
			return ((uint64_t)code << 32ul) | (uint64_t)idx;
		}

		__device__ inline uint2 determineRange(uint32_t const* mortonCodes,
			const uint32_t numObjs, uint32_t idx) {

			// If root, cover all
			if (idx == 0)
				return make_uint2(0, numObjs - 1);

			// Determine direction of the range
			const uint64_t selfCode = mergeIdx(mortonCodes[idx], idx); // [ leafMorton[Idx] | node[Idx] ]
			const int lDelta = commonUpperBits(selfCode, mergeIdx(mortonCodes[idx - 1], idx - 1));
			const int rDelta = commonUpperBits(selfCode, mergeIdx(mortonCodes[idx + 1], idx + 1));
			const int d = (rDelta > lDelta) ? 1 : -1;

			// Compute upper bound for the length of the range
			const int minDelta = thrust::min(lDelta, rDelta);
			int lMax = 2;
			int i;
			while ((i = idx + d * lMax) >= 0 && i < numObjs) {
				if (commonUpperBits(selfCode, mergeIdx(mortonCodes[i], i)) <= minDelta) break;
				lMax <<= 1;
			}

			// Find the exact range by binary search
			int t = lMax >> 1;
			int l = 0;
			while (t > 0) {
				i = idx + (l + t) * d;
				if (0 <= i && i < numObjs)
					if (commonUpperBits(selfCode, mergeIdx(mortonCodes[i], i)) > minDelta)
						l += t;
				t >>= 1;
			}

			unsigned int jdx = idx + l * d;
			if (d < 0) thrust::swap(idx, jdx); // Make sure that idx < jdx
			return make_uint2(idx, jdx);
		}

		__device__ inline uint32_t findSplit(uint32_t const* mortonCodes,
			const uint32_t first, const uint32_t last) {

			const uint64_t firstCode = mergeIdx(mortonCodes[first], first);
			const uint64_t lastCode = mergeIdx(mortonCodes[last], last);
			const int deltaNode = commonUpperBits(firstCode, lastCode);

			// Binary search for split position
			int split = first;
			int stride = last - first;
			do {
				stride = (stride + 1) >> 1;
				const int middle = split + stride;
				if (middle < last)
					if (commonUpperBits(firstCode, mergeIdx(mortonCodes[middle], middle)) > deltaNode)
						split = middle;
			} while (stride > 1);

			return split;
		}

		/**
		 * Builds out the internal nodes of the LBVH
		 * @param nodeRanges : node's full covering range for leaves
		 * @param clusterTriMax
		 * @param clusterTriMin
		 */
		__global__ void lbvhBuildInternalKernel(LBVH::node* nodes, uint32_t* leafParents,
			uint32_t const* mortonCodes, uint32_t const* objIDs, int numObjs, uint2* nodeRanges, const uint16_t clusterTriMax, const uint16_t
			clusterTriMin) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs - 1) return;

			// determine node's own range
			uint2 range = determineRange(mortonCodes, numObjs, tid);
			uint2 nodeRange = range;

			// Cluster things
			{
				uint32_t numCoveringTriangles = nodeRange.y - nodeRange.x + 1;
				bool isValidCluster = (clusterTriMin <= numCoveringTriangles) && (numCoveringTriangles <= clusterTriMax);
				// If Valid length(numTri), Mark on range's min
				MARK_MSB_UINT32(nodeRange.x, isValidCluster);
				nodeRanges[tid] = nodeRange; // write
			}


			nodes[tid].fence = (tid == range.x) ? range.y : range.x;
			const int gamma = findSplit(mortonCodes, range.x, range.y);

			// Left and right children are neighbors to the split point
			// Check if there are leaf nodes, which are indexed behind the (numObj - 1) internal nodes
			if (range.x == gamma) {
				leafParents[gamma] = (uint32_t)tid;
				range.x = gamma | 0x80000000;
			}
			else {
				range.x = gamma;
				nodes[range.x].parentIdx = (uint32_t)tid;
			}

			if (range.y == gamma + 1) {
				leafParents[gamma + 1] = (uint32_t)tid | 0x80000000;
				range.y = (gamma + 1) | 0x80000000;
			}
			else {
				range.y = gamma + 1;
				nodes[range.y].parentIdx = (uint32_t)tid | 0x80000000;
			}

			nodes[tid].leftIdx = range.x;
			nodes[tid].rightIdx = range.y;


		}

		// Refits the AABBs of the internal nodes
		__global__ void mergeUpKernel(LBVH::node* nodes,
			uint32_t* leafParents, AABB* aabbs, uint32_t* objIDs, int* flags, int numObjs) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs) return;

			// Keep track of the maximum stack size required for DFS
			// Assuming full exploration and always pushing the left child first.
			int depth = 1;

			AABB last = aabbs[objIDs[tid]];
			int parent = leafParents[tid];

			while (true) {
				int isRight = (parent & 0x80000000) != 0;
				parent = parent & 0x7FFFFFFF;
				nodes[parent].bounds[isRight] = last;

				// Exit if we are the first thread here
				int otherDepth = atomicOr(flags + parent, depth);
				if (!otherDepth) return;

				if (isRight)
					depth = max(depth + 1, otherDepth);
				else
					depth = max(depth, otherDepth + 1);

				// Ensure memory coherency before we read.
				__threadfence();

				if (!parent) {			// We've reached the root.
					flags[0] = depth;	// Only the one lucky thread gets to finish up.
					return;
				}
				last.absorb(nodes[parent].bounds[1 - isRight]);
				parent = nodes[parent].parentIdx;
			}
		}


		__device__ __forceinline__ int32_t GetCluterID(int triID, const uint32_t* leafParents, const LBVH::node* nodes, const uint2* nodeRanges)
		{
			// start from leaf's parent
			uint32_t curNode = IGNORE_MSB_UINT32(leafParents[triID]);
			int32_t retClusterID = -1; // -1 If Not included in cluster
			// bottom-up Find Cluster Node
			while (true)
			{
				uint2 nodeRange = nodeRanges[curNode];

				// Else If curNode is Valid, clusterID is curNode and return;
				if (UNPACK_MSB_UINT32(nodeRange.x) == 1)
				{
					retClusterID = curNode;
					curNode = IGNORE_MSB_UINT32(nodes[curNode].parentIdx);
					continue;
				}
				return retClusterID;
			}
		}
		template<uint16_t CLUSTER_TRI_MAX = 64>
		__global__ void Assign_ClusterID(int numTriangles, const uint32_t* __restrict__ leafParents,
			const LBVH::node* nodes, const uint2* nodeRanges, uint32_t* __restrict__  clusterCounter,
			const uint32_t* __restrict__ oldIndexBuffer, uint32_t* __restrict__ newIndexBuffer)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;

			if (tid >= numTriangles) return;

			int clusterID = GetCluterID(tid, leafParents, nodes, nodeRanges);


		}

		__global__ void Fill_ClusteredIndexBuffer_SimpleMorton(const uint32_t* __restrict__  triIDs, const uint32_t* __restrict__ oldIndexBuffer, uint32_t* __restrict__ newIndexBuffer, int numTriangles)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numTriangles) return;

			uint32_t triID = triIDs[tid];
			uint3 triangle = make_uint3(oldIndexBuffer[triID * 3 + 0], oldIndexBuffer[triID * 3 + 1], oldIndexBuffer[triID * 3 + 2]);
			newIndexBuffer[tid * 3 + 0] = triangle.x;
			newIndexBuffer[tid * 3 + 1] = triangle.y;
			newIndexBuffer[tid * 3 + 2] = triangle.z;
		}

		template<uint16_t CLUSTER_TRI_MAX = 64>
		__global__ void Fill_ClusteredIndexBuffer_BAD(const uint32_t* __restrict__  triIDs, int numTriangles, const uint32_t* __restrict__ leafParents,
			const LBVH::node* nodes, const uint2* nodeRanges, uint32_t* __restrict__  clusterCounter,
			const uint32_t* __restrict__ oldIndexBuffer, uint32_t* __restrict__ newIndexBuffer)
		{
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;

			if (tid >= numTriangles) return;

			int clusterID = GetCluterID(tid, leafParents, nodes, nodeRanges);
			if (clusterID < 0) return;

			uint8_t laneID = threadIdx.x & 31;
			uint32_t activeMask = __activemask(); // get active threads from cur warp
			uint32_t commonClusterGroup = __match_any_sync(activeMask, clusterID); // peer: thread mask with same clusterID
			uint32_t laneMask = (1 << laneID) - 1; // [ 0,,0 curIdxInWarp(0) 1 1,,1 1 ]
			// rank from the group. Count how many other lanes infront of me
			int localRank = __popc(commonClusterGroup & laneMask); // __popc : count 1.
			// leader : The Lowest Index
			int leaderLane = __ffs(commonClusterGroup) - 1; // __ffs : [10101,,,01100"1"00] -> 3, and 2 is index
			int groupSize = __popc(commonClusterGroup); // count commonClusterGroup
			int baseLocalOffset = 0; // offset from current cluster triangles

			if (laneID == leaderLane)
			{
				// Set baseOffset for this group(threads with common cluster), and add counter for next
				baseLocalOffset = atomicAdd(&clusterCounter[clusterID], groupSize);
			}

			// Broadcast baseOffset to the group peers
			baseLocalOffset = __shfl_sync(commonClusterGroup, baseLocalOffset, leaderLane);
			int finalDstOffset = ((clusterID * CLUSTER_TRI_MAX) + (baseLocalOffset + localRank)) * 3; // stride * 3

			uint3 tri = make_uint3(oldIndexBuffer[tid * 3 + 0], oldIndexBuffer[tid * 3 + 1], oldIndexBuffer[tid * 3 + 2]);
			((uint3*)newIndexBuffer)[finalDstOffset] = tri;
		}
	}


	LBVH::LBVH() : impl(std::make_unique<thrustImpl>())
	{
	}
	LBVH::~LBVH()
	{
	}

	void LBVH::compute(aabb* devicePtr, size_t size, uint16_t clusterSize, uint32_t* d_oldIndexBuffer, uint32_t* d_newIndexBuffer)
	{
		impl->d_objs = thrust::device_ptr<aabb>(devicePtr);
		numObjs = size;

		const unsigned int numInternalNodes = numObjs - 1;	// Total number of internal nodes, (binary tree)
		const unsigned int numNodes = numObjs * 2 - 1;		// Total number of nodes (leaves + internaNodes)

		impl->d_morton.resize(numObjs);
		impl->d_objIDs.resize(numObjs);
		impl->d_leafParents.resize(numObjs);
		impl->d_nodes.resize(numInternalNodes);
		impl->d_flags.resize(numInternalNodes);
		impl->d_nodeRanges.resize(numInternalNodes);
		impl->d_clusterTriCounter.resize(clusterSize);


		// Compute the bounding box for the whole scene so we can assign morton codes
		rootBounds = aabb();
		rootBounds = thrust::reduce(
			impl->d_objs, impl->d_objs + numObjs, rootBounds,
			[] __host__ __device__(const aabb & lhs, const aabb & rhs) {
			auto b = lhs;
			b.absorb(rhs);
			return b;
		});

		// Compute morton codes. These don't have to be unique here.
		Kernels::mortonKernel << <(numObjs + 255) / 256, 256 >> > (
			devicePtr, thrust::raw_pointer_cast(impl->d_morton.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()), rootBounds, numObjs);


		// Sort morton codes
		thrust::stable_sort_by_key(impl->d_morton.begin(), impl->d_morton.end(), impl->d_objIDs.begin());

		constexpr uint16_t clusterTriMax = 64;
		constexpr uint16_t clusterTriMin = (clusterTriMax >> 1) + 1;
		// Build out the internal nodes
		Kernels::lbvhBuildInternalKernel << <(numInternalNodes + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_morton.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			numObjs,
			thrust::raw_pointer_cast(impl->d_nodeRanges.data()), clusterTriMax, clusterTriMin);
		CUDA_SYNC_CHECK();

		Kernels::Fill_ClusteredIndexBuffer_BAD << <(numInternalNodes + 511) / 512, 512 >> > (
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			numObjs,
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_nodeRanges.data()),
			thrust::raw_pointer_cast(impl->d_clusterTriCounter.data()),
			d_oldIndexBuffer,
			d_newIndexBuffer);
		CUDA_SYNC_CHECK();
		CUDA_SYNC_CHECK();
		CUDA_SYNC_CHECK();

#if (1) // Thrust Stream Compaction

#else // CUB Stream Compaction
#endif

#if (0) // OLD
		constexpr uint32_t clusterMaxTriangles = 64;
		thrust::device_vector<uint32_t> d_clusterObjs(numObjs * clusterMaxTriangles, 0);


		// Collect Valid nodes (Device Compaction)
		thrust::device_vector<int> d_tempIndices(numInternalNodes);
		thrust::device_vector<int> d_validNodeIndices(numInternalNodes);
		thrust::sequence(d_tempIndices.begin(), d_tempIndices.end()); // [0,1,2,,,, N-1]
		int h_validNum, * d_validNum;
		cudaMalloc(&d_validNum, sizeof(int));
		void* d_tempStorage = nullptr;
		size_t tempStorageBytes = 0;

		IsValidNodeFunctor isValid = IsValidNodeFunctor(thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_nodeRanges.data()));

		cub::DeviceSelect::If(
			d_tempStorage,
			tempStorageBytes,
			d_tempIndices.begin(),
			d_validNodeIndices.begin(),
			d_validNum,
			(int)numInternalNodes,
			isValid);
		CUDA_SYNC_CHECK();
		cudaMalloc(&d_tempStorage, tempStorageBytes);
		cudaMemcpy(&h_validNum, d_validNum, sizeof(int), cudaMemcpyDeviceToHost);

		cub::DeviceSelect::If(
			d_tempStorage,
			tempStorageBytes,
			d_tempIndices.begin(),
			d_validNodeIndices.begin(),
			d_validNum,
			(int)numInternalNodes,
			isValid);
		CUDA_SYNC_CHECK();
		cudaMemcpy(&h_validNum, d_validNum, sizeof(int), cudaMemcpyDeviceToHost);


		// d_validNodeIndices : 2,23,34,,,, : d_validNum
		// getRange( nodes[d_validNodeIndices]  ). 

		// DEBUG
		thrust::host_vector<int> h_ValidNodeIndices(d_validNodeIndices.begin(), d_validNodeIndices.begin() + h_validNum);
		std::vector<int> vec(h_ValidNodeIndices.begin(), h_ValidNodeIndices.end());

		std::vector<int> nums;
		thrust::host_vector<uint2> hRanges(impl->d_nodeRanges);
		for (int i = 0; i < h_validNum; ++i)
		{
			uint2 range = hRanges[h_ValidNodeIndices[i]];
			int num = range.y - IGNORE_MSB_UINT32(range.x) + 1;
			nums.push_back(num);
		}
		std::sort(nums.begin(), nums.end());
		//cub::DeviceSelect::Flagged(d_tempStorage, tempStorageBytes,
		//	thrust::make_counting_iterator(0),   // [0, 1, 2, 3, ...]
		//	d_validMask,                         // Flag array (0/1)
		//	d_validNodeIndices,
		//	d_numValid,
		//	N)  
#pragma endregion



		//LBVHKernels::MakeClusters << <(numInternalNodes + 255) / 256, 256 >> > (
		//	thrust::raw_pointer_cast(impl->d_nodes.data()),
		//	thrust::raw_pointer_cast(impl->d_nodeRanges.data()),
		//	thrust::raw_pointer_cast(impl->d_objIDs.data()),
		//	numObjs,
		//	thrust::raw_pointer_cast(d_clusterObjs.data()), nullptr, nullptr);
		CUDA_CHECK_LAST_ERROR();

		std::vector<int> h_std_vec(numObjs * clusterMaxTriangles);
		thrust::copy(d_clusterObjs.begin(), d_clusterObjs.end(), h_std_vec.begin());
		thrust::sort(h_std_vec.begin(), h_std_vec.end());

		std::vector <LBVH::node> h_nodes(impl->d_nodes.size());
		thrust::copy(impl->d_nodes.begin(), impl->d_nodes.end(), h_nodes.begin());


		//auto diffChecker = [&](LBVH::node* pNode, uint32_t nodeID)
		//	{
		//		uint32_t ret = (h_std_vec[nodeID] - max(h_std_vec[pNode->leftIdx], h_std_vec[pNode->rightIdx]));

		//		diffChecker();
		//	};
		//diffChecker()
#endif
		CUDA_SYNC_CHECK();
	}


	void ClusterBuilder::Impl::Init(float* positions, uint32_t numPositions, uint32_t* indices, uint32_t numIndices)
	{
		assert(numIndices % 3 == 0);
		m_numTriangles = (numIndices / 3);

		// Device Alloc
		CUDA_CHECK(cudaMalloc(&m_dPositions, sizeof(float3) * numPositions));
		CUDA_CHECK(cudaMalloc(&m_dOldIndexBuffer, sizeof(uint3) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dNewIndexBuffer, sizeof(uint3) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dAABBs, sizeof(AABB*) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dMortons, sizeof(uint32_t*) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dTriIDs, sizeof(uint32_t*) * m_numTriangles));

		// Assign, Memcpy
		float3* f3Positions = (float3*)positions;
		CUDA_CHECK(cudaMemcpy(m_dPositions, f3Positions, sizeof(float3) * numPositions, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(m_dOldIndexBuffer, indices, sizeof(uint3) * m_numTriangles, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemset(m_dNewIndexBuffer, 0xFFFF'FFFF, sizeof(uint3) * m_numTriangles));


		// Fill AABB
		dim3 gridDim = dim3(ROUND_UP_DIM(m_numTriangles, 1024), 1, 1);
		/*gmcCuda::Kernels::Fill_AABBs << <gridDim, 1024 >> > (
			m_numTriangles,
			m_dPositions,
			(uint3*)m_dOldIndexBuffer,
			m_dAABBs);*/
		CUDA_SYNC_CHECK();
	}

	void ClusterBuilder::Impl::Init_WithExternalMappedMemory(float* mappedPositions, uint32_t numPositions,
		uint32_t* mappedIndices, uint32_t numIndices)
	{
		assert(numIndices % 3 == 0);
		m_numTriangles = (numIndices / 3);

		// Device Alloc
		//CUDA_CHECK(cudaMalloc(&m_dPositions, sizeof(float3) * numPositions));
		CUDA_CHECK(cudaMalloc(&m_dNewIndexBuffer, sizeof(uint3) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dAABBs, sizeof(AABB) * (m_numTriangles + 1))); // last is Root AABB
		CUDA_CHECK(cudaMalloc(&m_dMortons, sizeof(uint32_t*) * m_numTriangles));
		CUDA_CHECK(cudaMalloc(&m_dTriIDs, sizeof(uint32_t*) * m_numTriangles));

		// Map, Assign, Memcpy
		m_dPositions = (float3*)mappedPositions;
		m_dOldIndexBuffer = mappedIndices;

		CUDA_CHECK(cudaMemset(m_dNewIndexBuffer,  0xFFFF'FFFF, sizeof(uint3) * m_numTriangles));

		// Warming up
		{
			uint32_t blockSize = 256;
			dim3 gridDim = dim3(ROUND_UP_DIM(m_numTriangles, blockSize), 1, 1);
			gmcCuda::Kernels::Fill_AABBs << <gridDim, blockSize >> > (
				m_numTriangles,
				m_dPositions,
				m_dOldIndexBuffer,
				m_dAABBs);
			CUDA_SYNC_CHECK();
			m_rootAABB = thrust::reduce(
				thrust::device,
				m_dAABBs, m_dAABBs + m_numTriangles, m_rootAABB,
				Kernels::MergeAABBFunctor());
			CUDA_SYNC_CHECK();
			Kernels::mortonKernel << <gridDim, blockSize >> > (
				m_dAABBs, m_dMortons, m_dTriIDs, m_rootAABB, m_numTriangles);
			CUDA_SYNC_CHECK();
		}
	}

	void ClusterBuilder::Impl::BuildClusters()
	{

		// First Build
		if (m_dOldIndexBuffer == nullptr && m_dNewIndexBuffer == nullptr)
		{
			cudaMalloc(&m_dOldIndexBuffer, sizeof(uint32_t) * m_numTriangles * 3);
			uint32_t allocSize = sizeof(uint32_t) * ((m_numTriangles + clusterSize - 1) & ~clusterSize) * 3;
			cudaMalloc(&m_dNewIndexBuffer, allocSize); // multiple of clusterSize
			cudaMemset(m_dNewIndexBuffer, 0xFFFF'FFFF, allocSize);
		}

		// Build BVH
		LBVH bvh;
		printf("Building LBVH...\n");
		bvh.compute(m_dAABBs, m_numTriangles, clusterSize ,m_dOldIndexBuffer, m_dNewIndexBuffer);
		CUDA_SYNC_CHECK();



	}

	uint32_t ClusterBuilder::Impl::BuildClusters_SimpleMorton(uint16_t clusterMaxSize, gmc::Cluster* outClusters)
	{
		//gmcCuda::GPUTimer gpuTimer(5, 0);
		gmcCuda::GPUTimer gpuTimer(4, 0);
		cudaError_t err;

		// Fill AABB
		uint32_t blockSize = 256;
		dim3 gridDim = dim3(ROUND_UP_DIM(m_numTriangles, blockSize), 1, 1);

		gpuTimer.RecordStart();
		gmcCuda::Kernels::Compute_AABBs << <gridDim, blockSize >> > (
			m_numTriangles,
			m_dPositions,
			m_dOldIndexBuffer,
			m_dAABBs);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

		// Compute morton codes.
		gpuTimer.RecordStart();
		Kernels::mortonKernel << <gridDim, blockSize >> > (
		m_dAABBs, m_dMortons, m_dTriIDs, m_numTriangles);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

		//gpuTimer.RecordStart();
		//gmcCuda::Kernels::Fill_AABBs << <gridDim, blockSize >> > (
		//	m_numTriangles,
		//	m_dPositions,
		//	m_dOldIndexBuffer,
		//	m_dAABBs);
		//gpuTimer.RecordEnd();
		//CUDA_SYNC_CHECK();

		//// Root AABB
		//m_rootAABB = AABB();
		//gpuTimer.RecordStart();
		//m_rootAABB = thrust::reduce(
		//	thrust::device,
		//	m_dAABBs, m_dAABBs + m_numTriangles, m_rootAABB,
		//	Kernels::MergeAABBFunctor());
		//gpuTimer.RecordEnd();
		//CUDA_SYNC_CHECK();

		//// Compute morton codes.
		//gpuTimer.RecordStart();
		//Kernels::mortonKernel << <gridDim, blockSize >> > (
		//	m_dAABBs, m_dMortons, m_dTriIDs, m_rootAABB, m_numTriangles);
		//gpuTimer.RecordEnd();
		//CUDA_SYNC_CHECK();

		// Sort morton codes
		gpuTimer.RecordStart();
		thrust::stable_sort_by_key(thrust::device, m_dMortons, m_dMortons + m_numTriangles, m_dTriIDs);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

		gpuTimer.RecordStart();
		Kernels::Fill_ClusteredIndexBuffer_SimpleMorton<<<gridDim, blockSize >>> (m_dTriIDs, m_dOldIndexBuffer, m_dNewIndexBuffer, m_numTriangles);
		gpuTimer.RecordEnd();
		CUDA_SYNC_CHECK();

		gpuTimer.printResults();

		if (!m_bClustersCreated && cudaGetLastError() == cudaSuccess)
		{
			m_bClustersCreated = true;
			// make numClusters multiple of clusterMaxSize
			m_numClusters = (m_numTriangles + clusterMaxSize - 1) / clusterMaxSize;

			for (uint32_t i = 0; i < m_numClusters - 1; ++i)
			{
				outClusters[i] = gmc::Cluster
				{
					0, clusterMaxSize * i, clusterMaxSize * 3u, clusterMaxSize
				};
			}
			// last cluster's triangle count might be less than clusterMaxSize
			uint16_t remainTriangles = m_numTriangles & (clusterMaxSize - 1);
			if (remainTriangles == 0)
			{
				m_numClusters -= 1;
			}
			else
			{
				outClusters[m_numClusters - 1] = gmc::Cluster
				{
					0, clusterMaxSize * (m_numClusters - 1), clusterMaxSize * 3u, remainTriangles
				};				
			}
		}

		//CUDA_CHECK(cudaMemcpy(m_dOldIndexBuffer, m_dNewIndexBuffer, sizeof(uint3) * m_numTriangles, cudaMemcpyDeviceToDevice));
		std::swap(m_dOldIndexBuffer, m_dNewIndexBuffer);
		return m_numClusters;
	}
}
