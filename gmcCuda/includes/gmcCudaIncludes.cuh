#pragma once
/**
 * Include only for .cu, No .cpp
 * struct AABBare from KittenGpuLBVH(https://github.com/jerry060599/KittenGpuLBVH)'s Bound 
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

#pragma nv_diag_suppress esa_on_defaulted_function_ignored
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <cudaHelper.cuh>
#include <vec_math.h>


// Thrust
#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>

//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>
//#include <glm/gtc/matrix_access.hpp>
//#define GLM_ENABLE_EXPERIMENTAL
//#include <glm/gtx/compatibility.hpp>
//#include <glm/gtx/norm.hpp>

#include <string>
#include <functional>
#include <tuple>

namespace gmcCuda {
	//using namespace glm;
	//using namespace std;

	inline void print(int x, const char* format = "%d") {
		printf(format, x);
	}

	__device__ __host__ inline void printDiv(const char* label = nullptr) {
		if (!label) label = "div";
		printf("\n----%s----\n", label);
	}

	__device__ __host__ inline void print(float x, const char* format = "%f\n") {
		printf(format, x);
	}

	__device__ __host__ inline void print(double x, const char* format = "%f\n") {
		printf(format, x);
	}

	__host__ __device__
		void print(const float4 m[4], const char* format = "%.4f") {
		for (int i = 0; i < 4; ++i) {
			printf(i == 0 ? "{{" : " {");
			for (int j = 0; j < 4; ++j) {
				printf(format, (&m[j].x)[i]); // column-major, m[j] = column
				if (j != 3) printf(", ");
			}
			printf(i == 3 ? "}}\n" : "}\n");
		}
	}

	__host__ __device__
		void print(const float2& v, const char* format = "%.4f") {
		printf("{");
		printf(format, v.x); printf(", ");
		printf(format, v.y);
		printf("}\n");
	}

	__host__ __device__
		void print(const float3& v, const char* format = "%.4f") {
		printf("{");
		printf(format, v.x); printf(", ");
		printf(format, v.y); printf(", ");
		printf(format, v.z);
		printf("}\n");
	}

	__host__ __device__
		void print(const float4& v, const char* format = "%.4f") {
		printf("{");
		printf(format, v.x); printf(", ");
		printf(format, v.y); printf(", ");
		printf(format, v.z); printf(", ");
		printf(format, v.w);
		printf("}\n");
	}

	__host__ __device__
		void print(const int2& v, const char* format = "%d") {
		printf("{%d, %d}\n", v.x, v.y);
	}

	__host__ __device__
		void print(const int3& v, const char* format = "%d") {
		printf("{%d, %d, %d}\n", v.x, v.y, v.z);
	}

	__host__ __device__
		void print(const int4& v, const char* format = "%d") {
		printf("{%d, %d, %d, %d}\n", v.x, v.y, v.z, v.w);
	}

	__host__ __device__
		float3 MinFloat3(const float3& a, const float3& b) {
		return make_float3(
			fminf(a.x, b.x),
			fminf(a.y, b.y),
			fminf(a.z, b.z)
		);
	}
	__host__ __device__
		float Min(const float3& f3) {
		float m = f3.x;
		m = fminf(m, f3.y);
		m = fminf(m, f3.z);
		return m;
	}


	__host__ __device__
		float3 MaxFloat3(const float3& a, const float3& b) {
		return make_float3(
			fmaxf(a.x, b.x),
			fmaxf(a.y, b.y),
			fmaxf(a.z, b.z)
		);
	}
	__host__ __device__
		float Max(const float3& f3) {
		float m = f3.x;
		m = fmaxf(m, f3.y);
		m = fmaxf(m, f3.z);
		return m;
	}


	__device__ __host__ inline uint32_t mortonExpandBits(uint32_t v) {
		v = (v * 0x00010001u) & 0xFF0000FFu;
		v = (v * 0x00000101u) & 0x0F00F00Fu;
		v = (v * 0x00000011u) & 0xC30C30C3u;
		v = (v * 0x00000005u) & 0x49249249u;
		return v;
	}

	__device__ __host__ inline uint32_t getMorton(uint3 cell) {
		const uint32_t xx = mortonExpandBits(cell.x);
		const uint32_t yy = mortonExpandBits(cell.y);
		const uint32_t zz = mortonExpandBits(cell.z);
		return xx * 4 + yy * 2 + zz;
	}

	struct AABB {
		float3 min;
		float3 max;

		__host__ __device__
			AABB()
			: min(make_float3(INFINITY, INFINITY, INFINITY)),
			max(make_float3(-INFINITY, -INFINITY, -INFINITY)) {
		}

		__host__ __device__
			AABB(const float3& center)
			: min(center), max(center) {
		}

		__host__ __device__
			AABB(const float3& min_, const float3& max_)
			: min(min_), max(max_) {
		}

		__host__ __device__
			AABB(const AABB& b)
			: min(b.min), max(b.max) {
		}

		__host__ __device__
			float3 center() const {
			return make_float3(
				0.5f * (min.x + max.x),
				0.5f * (min.y + max.y),
				0.5f * (min.z + max.z)
			);
		}

		__host__ __device__
			void absorb(const AABB& b) {
			min.x = fminf(min.x, b.min.x); min.y = fminf(min.y, b.min.y); min.z = fminf(min.z, b.min.z);
			max.x = fmaxf(max.x, b.max.x); max.y = fmaxf(max.y, b.max.y); max.z = fmaxf(max.z, b.max.z);
		}

		__host__ __device__
			void absorb(const float3& p) {
			min.x = fminf(min.x, p.x); min.y = fminf(min.y, p.y); min.z = fminf(min.z, p.z);
			max.x = fmaxf(max.x, p.x); max.y = fmaxf(max.y, p.y); max.z = fmaxf(max.z, p.z);
		}

		__host__ __device__
			bool contains(const float3& p) const {
			return (min.x <= p.x && p.x <= max.x) &&
				(min.y <= p.y && p.y <= max.y) &&
				(min.z <= p.z && p.z <= max.z);
		}

		__host__ __device__
			bool contains(const AABB& b) const {
			return contains(b.min) && contains(b.max);
		}

		__host__ __device__
			bool intersects(const AABB& b) const {
			return !(max.x < b.min.x || min.x > b.max.x ||
				max.y < b.min.y || min.y > b.max.y ||
				max.z < b.min.z || min.z > b.max.z);
		}

		__host__ __device__
			void pad(float padding) {
			min.x -= padding; min.y -= padding; min.z -= padding;
			max.x += padding; max.y += padding; max.z += padding;
		}

		__host__ __device__
			void pad(const float3& padding) {
			min.x -= padding.x; min.y -= padding.y; min.z -= padding.z;
			max.x += padding.x; max.y += padding.y; max.z += padding.z;
		}

		__host__ __device__
			float volume() const {
			float dx = max.x - min.x;
			float dy = max.y - min.y;
			float dz = max.z - min.z;
			return dx * dy * dz;
		}

		__host__ __device__
			float3 normCoord(const float3& pos) const {
			return make_float3(
				(pos.x - min.x) / (max.x - min.x),
				(pos.y - min.y) / (max.y - min.y),
				(pos.z - min.z) / (max.z - min.z)
			);
		}

		__host__ __device__
			float3 interp(const float3& coord) const {
			return make_float3(
				min.x + (max.x - min.x) * coord.x,
				min.y + (max.y - min.y) * coord.y,
				min.z + (max.z - min.z) * coord.z
			);
		}
	};

}

namespace std {
	namespace {

		// Code from boost
		// Reciprocal of the golden ratio helps spread entropy
		//     and handles duplicates.
		// See Mike Seymour in magic-numbers-in-boosthash-combine:
		//     http://stackoverflow.com/questions/4948780

		template <class T>
		inline void hash_combine(std::size_t& seed, T const& v) {
			seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}

		// Recursive template code derived from Matthieu M.
		template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
		struct HashValueImpl {
			static void apply(size_t& seed, Tuple const& tuple) {
				HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
				hash_combine(seed, std::get<Index>(tuple));
			}
		};

		template <class Tuple>
		struct HashValueImpl<Tuple, 0> {
			static void apply(size_t& seed, Tuple const& tuple) {
				hash_combine(seed, std::get<0>(tuple));
			}
		};
	}

	//template <int N>
	//struct hash<glm::vec<N, int, glm::defaultp>> {
	//	std::size_t operator() (const glm::vec<N, int, glm::defaultp>& v) const {
	//		size_t h = 0x9e3779b9;
	//		for (int i = 0; i < N; i++)
	//			h = v[i] ^ (h + 0x9e3779b9 + (v[i] << 6) + (v[i] >> 2));
	//		return h;
	//	}
	//};

	//template <int N>
	//struct hash<glm::vec<N, glm::i64, glm::defaultp>> {
	//	std::size_t operator() (const glm::vec<N, int, glm::defaultp>& v) const {
	//		size_t h = 0x9e3779b9;
	//		for (int i = 0; i < N; i++)
	//			h = v[i] ^ (h + 0x9e3779b9llu + (v[i] << 6) + (v[i] >> 2));
	//		return h;
	//	}
	//};
};