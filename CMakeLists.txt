cmake_minimum_required(VERSION 3.30)
project(PC_challenge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(PC_challenge main.cpp)

target_link_libraries(PC_challenge PRIVATE OpenMP::OpenMP_CXX)
