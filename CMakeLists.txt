cmake_minimum_required(VERSION 3.30)
project(PC_challenge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

file(
        GLOB_RECURSE mains RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)


foreach (mainfile IN LISTS mains)
    # Get file name without directory
    get_filename_component(mainname ${mainfile} NAME_WE)
    add_executable(${mainname} ${mainfile})
    target_link_libraries(${mainname} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${mainname} PRIVATE -O3)
endforeach ()