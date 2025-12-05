# 헤더 파일 목록 (VS 필터 구조 반영용)
file(GLOB HEADERS "*.h" "*.hpp")

# 인터페이스 라이브러리 생성 (소스 없이 헤더만 존재)
add_library(Includes INTERFACE)

# 인클루드 경로 노출 (다른 프로젝트가 이 라이브러리를 링크하면 자동으로 경로가 추가됨)
target_include_directories(Includes INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# IDE에서 파일 보이게 설정
target_sources(Includes INTERFACE ${HEADERS})