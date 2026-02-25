# 综合测试 CMakeLists.txt

# 添加综合测试可执行文件
add_executable(combined_test
    src/combined_test.cpp
    src/mm_reader.cpp
)

target_include_directories(combined_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(combined_test PRIVATE
    linear_solver
)

# 禁用 Boost 依赖（amgcl）
target_compile_definitions(combined_test PRIVATE AMGCL_NO_BOOST)

# 安装规则
install(TARGETS combined_test DESTINATION bin)
