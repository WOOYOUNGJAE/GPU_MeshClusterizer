#include <clusterBuilder.cuh>

#include <algorithm>

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


	namespace LBVHKernels {


		// Computes the morton codes for each AABB
		/**
		 * @param ids : threadGlobalID(TriID)
		 * @param size : num Triangles
		 */
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

			// This is the root node
			if (idx == 0)
				return make_uint2(0, numObjs - 1);

			// Determine direction of the range
			const uint64_t selfCode = mergeIdx(mortonCodes[idx], idx);
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

		// We do 128 wide blocks which gets us 8KB of shared memory per block on compute 8.6. 
		// Ideally leave some left for L1 cache.
		constexpr int MAX_RES_PER_BLOCK = 4 * 128;

		// Query the LBVH for overlapping objects
		// Overcomplicated because of shared memory buffering
		template<bool IGNORE_SELF, int STACK_SIZE>
		__global__ void lbvhQueryKernel(uint2* res, int* resCounter, int maxRes,
			const LBVH::node* nodes, const uint32_t* objIDs,
			const uint32_t* queryIDs, const AABB* queryAABBs, const int numQueries) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			bool active = tid < numQueries;

			AABB queryAABB;
			int objIdx;
			if (active) {
				objIdx = queryIDs[tid];
				queryAABB = queryAABBs[objIdx];
			}

			__shared__ uint2 sharedRes[MAX_RES_PER_BLOCK];
			__shared__ int sharedCounter;		// How many results are cached in shared memory
			__shared__ int sharedGlobalIdx;		// Where to write in global memory
			if (threadIdx.x == 0)
				sharedCounter = 0;

			uint32_t stack[STACK_SIZE];			// This is dynamically sized through templating
			uint32_t* stackPtr = stack;
			*(stackPtr++) = 0;					// Push

			while (true) {
				__syncthreads();

				if (active)
					while (stackPtr != stack) {
						uint32_t nodeIdx = *(--stackPtr);	// Pop
						bool isLeaf = nodeIdx & 0x80000000;
						nodeIdx = nodeIdx & 0x7FFFFFFF;

						if (isLeaf) {
							if (IGNORE_SELF)
								if (nodeIdx <= tid) continue;

							// Add to shared memory
							int sIdx = atomicAdd(&sharedCounter, 1);
							if (sIdx >= MAX_RES_PER_BLOCK) {
								// We cannot sync here so we push the node back on the stack and wait
								*(stackPtr++) = nodeIdx | 0x80000000;
								break;
							}
							sharedRes[sIdx] = make_uint2(objIDs[nodeIdx], objIdx);
						}
						else {
							auto node = nodes[nodeIdx];

							// Ignore duplicate and self intersections
							if (IGNORE_SELF)
								if (max(nodeIdx, node.fence) <= tid) continue;

							// Internal node
							if (node.bounds[0].intersects(queryAABB))
								*(stackPtr++) = node.leftIdx;	// Push

							if (node.bounds[1].intersects(queryAABB))
								*(stackPtr++) = node.rightIdx;	// Push
						}
					}

				// Flush whatever we have
				__syncthreads();
				int totalRes = min(sharedCounter, MAX_RES_PER_BLOCK);

				if (threadIdx.x == 0)
					sharedGlobalIdx = atomicAdd(resCounter, totalRes);

				__syncthreads();

				// Make sure we dont write out of bounds
				const int globalIdx = sharedGlobalIdx;

				if (globalIdx >= maxRes || !totalRes) return;	// Out of memory for results.
				if (threadIdx.x == 0) sharedCounter = 0;

				// If we got here with a half full buffer, we are done.
				bool done = totalRes < MAX_RES_PER_BLOCK;
				// If we are about to run out of memory, we are done.
				if (totalRes > maxRes - globalIdx) {
					totalRes = maxRes - globalIdx;
					done = true;
				}

				// Copy full blocks
				int fullBlocks = (totalRes - 1) / (int)blockDim.x;
				for (int i = 0; i < fullBlocks; i++) {
					int idx = i * blockDim.x + threadIdx.x;
					res[globalIdx + idx] = sharedRes[idx];
				}

				// Copy the rest
				int idx = fullBlocks * blockDim.x + threadIdx.x;
				if (idx < totalRes) res[globalIdx + idx] = sharedRes[idx];

				// Break if every thread is done.
				if (done) break;
			}
		}

		// We are primarily limited by the number of registers, so we always call the kernel with just enough stack space.
		// This gives another ~15% performance boost for queries.
		template<bool IGNORE_SELF>
		void launchQueryKernel(uint2* res, int* resCounter, int maxRes,
			const LBVH::node* nodes, const uint32_t* objIDs,
			const uint32_t* queryIDs, const AABB* queryAABBs, const int numQueries, int stackSize) {

			// This is a bit ugly but we want to compile the kernel for all stack sizes.
#define DISPATCH_QUERY(N) case N: lbvhQueryKernel<IGNORE_SELF, N> << <(numQueries + 127) / 128, 128 >> > (res, resCounter, maxRes, nodes, objIDs, queryIDs, queryAABBs, numQueries); break;
			switch (stackSize) {
			default:
				DISPATCH_QUERY(32); DISPATCH_QUERY(31); DISPATCH_QUERY(30); DISPATCH_QUERY(29); DISPATCH_QUERY(28); DISPATCH_QUERY(27); DISPATCH_QUERY(26); DISPATCH_QUERY(25);
				DISPATCH_QUERY(24); DISPATCH_QUERY(23); DISPATCH_QUERY(22); DISPATCH_QUERY(21); DISPATCH_QUERY(20); DISPATCH_QUERY(19); DISPATCH_QUERY(18); DISPATCH_QUERY(17);
				DISPATCH_QUERY(16); DISPATCH_QUERY(15); DISPATCH_QUERY(14); DISPATCH_QUERY(13); DISPATCH_QUERY(12); DISPATCH_QUERY(11); DISPATCH_QUERY(10); DISPATCH_QUERY(9);
				DISPATCH_QUERY(8); DISPATCH_QUERY(7); DISPATCH_QUERY(6); DISPATCH_QUERY(5); DISPATCH_QUERY(4); DISPATCH_QUERY(3); DISPATCH_QUERY(2); DISPATCH_QUERY(1);
			}
#undef DISPATCH_QUERY
		}

		__device__ __forceinline__ int32_t GetCluterID(int triID, const uint32_t* leafParents, const LBVH::node* nodes, const uint2* nodeRanges)
		{
			// start from leaf's parent
			uint32_t curNode = leafParents[triID];
			int32_t retClusterID = -1; // -1 If Not included in cluster
			// bottom-up Find Cluster Node
			while (true)
			{
				uint2 nodeRange = nodeRanges[curNode];

				// Else If curNode is Valid, clusterID is curNode and return;
				if (UNPACK_MSB_UINT32(nodeRange.x) == 1)
				{
					retClusterID = curNode;
					curNode = nodes[curNode].parentIdx;
					continue;
				}
				return retClusterID;
			}
		}
		template<uint16_t CLUSTER_TRI_MAX = 64>
		__global__ void Fill_ClusteredIndexBuffer(const uint32_t* __restrict__  triIDs, int numTriangles, const uint32_t* __restrict__ leafParents,
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

			//// Broadcast baseOffset to the group peers
			//baseLocalOffset = __shfl_sync(commonClusterGroup, baseLocalOffset, leaderLane);
			//int finalDstOffset = ((clusterID * CLUSTER_TRI_MAX) + (baseLocalOffset + localRank)) * 3; // stride * 3

			//uint3 tri = make_uint3(oldIndexBuffer[tid * 3 + 0], oldIndexBuffer[tid * 3 + 1], oldIndexBuffer[tid * 3 + 2]);
			//((uint3*)newIndexBuffer)[finalDstOffset] = tri;
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
		LBVHKernels::mortonKernel << <(numObjs + 255) / 256, 256 >> > (
			devicePtr, thrust::raw_pointer_cast(impl->d_morton.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()), rootBounds, numObjs);


		// Sort morton codes
		thrust::stable_sort_by_key(impl->d_morton.begin(), impl->d_morton.end(), impl->d_objIDs.begin());

		constexpr uint16_t clusterTriMax = 64;
		constexpr uint16_t clusterTriMin = (clusterTriMax >> 1) + 1;
		// Build out the internal nodes
		LBVHKernels::lbvhBuildInternalKernel << <(numInternalNodes + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_morton.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			numObjs,
			thrust::raw_pointer_cast(impl->d_nodeRanges.data()), clusterTriMax, clusterTriMin);


		CUDA_SYNC_CHECK();

		LBVHKernels::Fill_ClusteredIndexBuffer << <(numInternalNodes + 511) / 512, 512 >> > (
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			numObjs,
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_nodeRanges.data()),
			thrust::raw_pointer_cast(impl->d_clusterTriCounter.data()),
			d_oldIndexBuffer,
			d_newIndexBuffer);
		CUDA_SYNC_CHECK();

//#if (1) // Thrust Stream Compaction
//
//#else // CUB Stream Compaction
//#endif
//
//#if (0) // OLD
//		constexpr uint32_t clusterMaxTriangles = 64;
//		thrust::device_vector<uint32_t> d_clusterObjs(numObjs * clusterMaxTriangles, 0);
//
//
//		// Collect Valid nodes (Device Compaction)
//		thrust::device_vector<int> d_tempIndices(numInternalNodes);
//		thrust::device_vector<int> d_validNodeIndices(numInternalNodes);
//		thrust::sequence(d_tempIndices.begin(), d_tempIndices.end()); // [0,1,2,,,, N-1]
//		int h_validNum, * d_validNum;
//		cudaMalloc(&d_validNum, sizeof(int));
//		void* d_tempStorage = nullptr;
//		size_t tempStorageBytes = 0;
//
//		IsValidNodeFunctor isValid = IsValidNodeFunctor(thrust::raw_pointer_cast(impl->d_nodes.data()),
//			thrust::raw_pointer_cast(impl->d_nodeRanges.data()));
//
//		cub::DeviceSelect::If(
//			d_tempStorage,
//			tempStorageBytes,
//			d_tempIndices.begin(),
//			d_validNodeIndices.begin(),
//			d_validNum,
//			(int)numInternalNodes,
//			isValid);
//		CUDA_SYNC_CHECK();
//		cudaMalloc(&d_tempStorage, tempStorageBytes);
//		cudaMemcpy(&h_validNum, d_validNum, sizeof(int), cudaMemcpyDeviceToHost);
//
//		cub::DeviceSelect::If(
//			d_tempStorage,
//			tempStorageBytes,
//			d_tempIndices.begin(),
//			d_validNodeIndices.begin(),
//			d_validNum,
//			(int)numInternalNodes,
//			isValid);
//		CUDA_SYNC_CHECK();
//		cudaMemcpy(&h_validNum, d_validNum, sizeof(int), cudaMemcpyDeviceToHost);
//
//
//		// d_validNodeIndices : 2,23,34,,,, : d_validNum
//		// getRange( nodes[d_validNodeIndices]  ). 
//
//		// DEBUG
//		thrust::host_vector<int> h_ValidNodeIndices(d_validNodeIndices.begin(), d_validNodeIndices.begin() + h_validNum);
//		std::vector<int> vec(h_ValidNodeIndices.begin(), h_ValidNodeIndices.end());
//
//		std::vector<int> nums;
//		thrust::host_vector<uint2> hRanges(impl->d_nodeRanges);
//		for (int i = 0; i < h_validNum; ++i)
//		{
//			uint2 range = hRanges[h_ValidNodeIndices[i]];
//			int num = range.y - IGNORE_MSB_UINT32(range.x) + 1;
//			nums.push_back(num);
//		}
//		std::sort(nums.begin(), nums.end());
//		//cub::DeviceSelect::Flagged(d_tempStorage, tempStorageBytes,
//		//	thrust::make_counting_iterator(0),   // [0, 1, 2, 3, ...]
//		//	d_validMask,                         // Flag array (0/1)
//		//	d_validNodeIndices,
//		//	d_numValid,
//		//	N)  
//#pragma endregion
//
//
//
//		//LBVHKernels::MakeClusters << <(numInternalNodes + 255) / 256, 256 >> > (
//		//	thrust::raw_pointer_cast(impl->d_nodes.data()),
//		//	thrust::raw_pointer_cast(impl->d_nodeRanges.data()),
//		//	thrust::raw_pointer_cast(impl->d_objIDs.data()),
//		//	numObjs,
//		//	thrust::raw_pointer_cast(d_clusterObjs.data()), nullptr, nullptr);
//		CUDA_CHECK_LAST_ERROR();
//
//		std::vector<int> h_std_vec(numObjs * clusterMaxTriangles);
//		thrust::copy(d_clusterObjs.begin(), d_clusterObjs.end(), h_std_vec.begin());
//		thrust::sort(h_std_vec.begin(), h_std_vec.end());
//
//		std::vector <LBVH::node> h_nodes(impl->d_nodes.size());
//		thrust::copy(impl->d_nodes.begin(), impl->d_nodes.end(), h_nodes.begin());
//
//
//		//auto diffChecker = [&](LBVH::node* pNode, uint32_t nodeID)
//		//	{
//		//		uint32_t ret = (h_std_vec[nodeID] - max(h_std_vec[pNode->leftIdx], h_std_vec[pNode->rightIdx]));
//
//		//		diffChecker();
//		//	};
//		//diffChecker()
//#endif
		CUDA_SYNC_CHECK();
	}


	void ClusterBuilder::Impl::BuildClusters()
	{
		uint32_t numTriangles = m_pGeometry->m_numTriangles;
		thrust::device_vector<uint2> d_res(100 * numTriangles);
		CUDA_SYNC_CHECK();


		// First Build
		if (m_dOldIndexBuffer == nullptr && m_dNewIndexBuffer == nullptr)
		{
			cudaMalloc(&m_dOldIndexBuffer, sizeof(uint32_t) * numTriangles * 3);
			uint32_t allocSize = sizeof(uint32_t) * ((numTriangles + clusterSize - 1) & ~clusterSize) * 3;
			cudaMalloc(&m_dNewIndexBuffer, allocSize); // multiple of clusterSize
			cudaMemset(m_dNewIndexBuffer, 0xFFFF'FFFF, allocSize);
		}

		// Build BVH
		LBVH bvh;
		printf("Building LBVH...\n");
		bvh.compute(thrust::raw_pointer_cast(m_pGeometry->m_dAABBs.data()), numTriangles, clusterSize ,m_dOldIndexBuffer, m_dNewIndexBuffer);
		CUDA_SYNC_CHECK();



	}

	
}