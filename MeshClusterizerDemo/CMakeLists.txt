# 1. 소스 파일 지정 (헤더/소스 자동 검색)
file(GLOB SOURCES "*.cpp" "*.h")

# 2. 실행 파일 생성
add_executable(MeshClusterizerDemo ${SOURCES})

# 3. 인클루드 경로 설정 (요청하신 경로 반영)
target_include_directories(MeshClusterizerDemo PRIVATE
    ${CMAKE_SOURCE_DIR}/External/glm        # $(SolutionDir)External\glm
    ${CMAKE_SOURCE_DIR}/External/tinygltf   # $(SolutionDir)External\tinygltf
    ${CMAKE_SOURCE_DIR}/Includes            # $(SolutionDir)Includes
)

# 4. 라이브러리 링크
# gmcCuda 프로젝트와 링크 (Demo가 gmcCuda.dll을 사용하므로)
target_link_libraries(MeshClusterizerDemo PRIVATE gmcCuda)