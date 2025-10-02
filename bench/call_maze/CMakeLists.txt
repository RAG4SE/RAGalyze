cmake_minimum_required(VERSION 3.16)
project(CallMaze LANGUAGES CXX)

add_executable(call_maze
    src/main.cpp
)

target_include_directories(call_maze PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(call_maze PRIVATE cxx_std_17)

