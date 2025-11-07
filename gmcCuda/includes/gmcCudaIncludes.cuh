#pragma once
/**
 * Include only for .cu, No .cpp
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

#pragma nv_diag_suppress esa_on_defaulted_function_ignored
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/norm.hpp>

#include <string>
#include <functional>
#include <tuple>

namespace gmcCuda {
	using namespace glm;
	using namespace std;

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

	template <int c, int r, typename T>
	__device__ __host__ void print(mat<c, r, T, defaultp> m, const char* format = "%.4f") {
		for (int i = 0; i < r; i++) {
			printf(i == 0 ? "{{" : " {");

			for (int j = 0; j < c; j++) {
				printf(format, m[j][i]);
				if (j != c - 1) printf(", ");
			}

			printf(i == r - 1 ? "}}\n" : "}\n");
		}
	}

	template <int s, typename T>
	__device__ __host__ void print(vec<s, T, defaultp> v, const char* format = "%.4f") {
		printf("{");
		for (int i = 0; i < s; i++) {
			printf(format, v[i]);
			if (i != s - 1) printf(", ");
		}
		printf("}\n");
	}

	template <int s>
	__device__ __host__ void print(vec<s, int, defaultp> v, const char* format = "%d") {
		printf("{");
		for (int i = 0; i < s; i++) {
			printf(format, v[i]);
			if (i != s - 1) printf(", ");
		}
		printf("}\n");
	}

	__device__ __host__ inline uint32_t mortonExpandBits(uint32_t v) {
		v = (v * 0x00010001u) & 0xFF0000FFu;
		v = (v * 0x00000101u) & 0x0F00F00Fu;
		v = (v * 0x00000011u) & 0xC30C30C3u;
		v = (v * 0x00000005u) & 0x49249249u;
		return v;
	}

	__device__ __host__ inline uint32_t getMorton(ivec3 cell) {
		const uint32_t xx = mortonExpandBits(cell.x);
		const uint32_t yy = mortonExpandBits(cell.y);
		const uint32_t zz = mortonExpandBits(cell.z);
		return xx * 4 + yy * 2 + zz;
	}

	template<int dim = 3, typename Real = float>
	struct Bound {
		vec<dim, Real, defaultp> min;
		vec<dim, Real, defaultp> max;

		__device__ __host__ Bound() :
			min(vec3(INFINITY)),
			max(vec3(-INFINITY)) {
		}

		__device__ __host__ Bound(vec<dim, Real, defaultp> center) : min(center), max(center) {}
		__device__ __host__ Bound(vec<dim, Real, defaultp> min, vec<dim, Real, defaultp> max) : min(min), max(max) {}

		__device__ __host__ Bound(const Bound<dim, Real>& b) : min(b.min), max(b.max) {}

		__device__ __host__ inline vec<dim, Real, defaultp> center() const {
			return (min + max) * 0.5f;
		}

		__device__ __host__ inline void absorb(const Bound<dim, Real>& b) {
			min = glm::min(min, b.min); max = glm::max(max, b.max);
		}

		__device__ __host__ inline void absorb(const vec<dim, Real, defaultp>& b) {
			min = glm::min(min, b); max = glm::max(max, b);
		}

		__device__ __host__ inline bool contains(const vec<dim, Real, defaultp>& point) const {
			return glm::all(glm::lessThanEqual(min, point)) && glm::all(glm::greaterThanEqual(max, point));
		}

		__device__ __host__ inline bool contains(const Bound<dim, Real>& b) const {
			return glm::all(glm::lessThanEqual(min, b.min)) && glm::all(glm::greaterThanEqual(max, b.max));
		}

		__device__ __host__ inline bool intersects(const Bound<dim, Real>& b) const {
			return !(glm::any(glm::lessThanEqual(max, b.min)) || glm::any(glm::greaterThanEqual(min, b.max)));
		}

		__device__ __host__ inline void pad(Real padding) {
			min -= vec<dim, Real, defaultp>(padding); max += vec<dim, Real, defaultp>(padding);
		}

		__device__ __host__ inline void pad(const vec<dim, Real, defaultp>& padding) {
			min -= padding; max += padding;
		}

		__device__ __host__ inline Real volume() const {
			vec<dim, Real, defaultp> diff = max - min;
			Real v = diff.x;
			for (int i = 1; i < dim; i++) v *= diff[i];
			return v;
		}

		__device__ __host__ inline vec<dim, Real, defaultp> normCoord(const vec<dim, Real, defaultp>& pos) const {
			return (pos - min) / (max - min);
		}

		__device__ __host__ inline vec<dim, Real, defaultp> interp(const vec<dim, Real, defaultp>& coord) const {
			vec<dim, Real, defaultp> pos;
			vec<dim, Real, defaultp> diff = max - min;
			for (int i = 0; i < dim; i++)
				pos[i] = min[i] + diff[i] * coord[i];
			return pos;
		}
	};

	template<typename Real>
	struct Bound<1, Real> {
		Real min;
		Real max;

		__device__ __host__ Bound() :
			min(INFINITY),
			max(-INFINITY) {
		}
		__device__ __host__ Bound(Real center) : min(center), max(center) {}
		__device__ __host__ Bound(Real min, Real max) : min(min), max(max) {}

		__device__ __host__ inline Real center() const {
			return (min + max) * 0.5f;
		}

		__device__ __host__ inline void absorb(const Bound<1, Real>& b) {
			min = glm::min(min, b.min); max = glm::max(max, b.max);
		}

		__device__ __host__ inline void absorb(Real b) {
			min = glm::min(min, b); max = glm::max(max, b);
		}

		__device__ __host__ inline bool contains(Real point) const {
			return min <= point && point <= max;
		}

		__device__ __host__ inline bool contains(const Bound<1, Real>& b) const {
			return min <= b.min && b.max <= max;
		}

		__device__ __host__ inline bool intersects(const Bound<1, Real>& b) const {
			return max > b.min && min < b.max;
		}

		__device__ __host__ inline void pad(Real padding) {
			min -= padding; max += padding;
		}

		__device__ __host__ inline Real volume() const {
			return max - min;
		}

		__device__ __host__ inline Real normCoord(Real pos) const {
			return (pos - min) / (max - min);
		}

		__device__ __host__ inline Real interp(Real coord) const {
			return min + (max - min) * coord;
		}
	};

	typedef Bound<1, float> Range;
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

	template <int N>
	struct hash<glm::vec<N, int, glm::defaultp>> {
		std::size_t operator() (const glm::vec<N, int, glm::defaultp>& v) const {
			size_t h = 0x9e3779b9;
			for (int i = 0; i < N; i++)
				h = v[i] ^ (h + 0x9e3779b9 + (v[i] << 6) + (v[i] >> 2));
			return h;
		}
	};

	template <int N>
	struct hash<glm::vec<N, glm::i64, glm::defaultp>> {
		std::size_t operator() (const glm::vec<N, int, glm::defaultp>& v) const {
			size_t h = 0x9e3779b9;
			for (int i = 0; i < N; i++)
				h = v[i] ^ (h + 0x9e3779b9llu + (v[i] << 6) + (v[i] >> 2));
			return h;
		}
	};
};
