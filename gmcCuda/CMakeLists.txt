# 소스/헤더 자동 수집
file(GLOB SOURCES src/*.cpp src/*.cu)
file(GLOB HEADERS_PUBLIC ${CMAKE_SOURCE_DIR}/Includes/*.h)
file(GLOB HEADERS_PUBLIC_CUDA ${CMAKE_SOURCE_DIR}/Includes/gmcCuda/*.h)
file(GLOB HEADERS_PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/includes/*.h ${CMAKE_CURRENT_SOURCE_DIR}/includes/*.cuh)
file(GLOB HEADERS_NVUTILS ${CMAKE_CURRENT_SOURCE_DIR}/includes/nvUtils/*.h)

# 라이브러리 생성
add_library(gmcCuda SHARED ${SOURCES} ${HEADERS_PUBLIC_CUDA} ${HEADERS_PRIVATE} ${HEADERS_NVUTILS})

# VS 필터 구조 복원
source_group("public headers" FILES ${HEADERS_PUBLIC})
source_group("public headers\\gmcCuda" FILES ${HEADERS_PUBLIC_CUDA})
source_group("private headers" FILES ${HEADERS_PRIVATE})
source_group("src" FILES ${SOURCES})
source_group("private headers\\nvUtils" FILES ${HEADERS_NVUTILS})


# 인클루드 경로 설정
target_include_directories(gmcCuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/nvUtils
    ${CMAKE_SOURCE_DIR}/External/glm  # 외부 GLM 라이브러리 경로
)

# Defines 설정 (DLL Export 매크로)
target_compile_definitions(gmcCuda PRIVATE GMC_EXPORTS)

# 라이브러리 링크 (Includes 프로젝트 및 CUDA 런타임)
target_link_libraries(gmcCuda PRIVATE Includes)
# CUDA는 LANGUAGES CUDA로 인해 기본 런타임이 자동 링크되지만, 명시가 필요하면 아래 주석 해제
# target_link_libraries(gmcCuda PRIVATE CUDA::cudart) 

# CUDA 컴파일 옵션 (extended-lambda 등 vcxproj 설정 반영)
target_compile_options(gmcCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)