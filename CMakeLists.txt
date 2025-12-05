cmake_minimum_required(VERSION 3.20)

# 프로젝트 및 언어 설정 (C++17, CUDA)
project(GPU_MeshClusterizer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 아키텍처 설정 (gmcCuda.vcxproj의 compute_70, sm_70 반영)
set(CMAKE_CUDA_ARCHITECTURES 70)

# 출력 디렉터리 설정 (bin/lib 폴더 구조 유지)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# 서브 프로젝트 추가
add_subdirectory(Includes)
add_subdirectory(gmcCuda)
add_subdirectory(MeshClusterizerDemo)