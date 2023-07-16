cmake_minimum_required(VERSION 3.15)

project(HttpServer)

include_directories(${CMAKE_SOURCE_DIR}/src/TrtLib)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.c *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

# link_directories(${CUDA_LIB_DIR}) 

# create static library
add_library(${PROJECT_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
