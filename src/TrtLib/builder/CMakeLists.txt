cmake_minimum_required(VERSION 3.15)
project(TrtBuilder LANGUAGES CUDA CXX)

# # 第三方库
set(CUDA_HOME /usr/local/cuda)
set(TRT_HOME /home/zwy/TensorRT-7.2.3.4)
enable_language(CUDA)

# # CUDA and cudnn include dir 
include_directories(${CUDA_HOME}/include)
include_directories(${CUDA_HOME}/targets/x86_64-linux/include)

# TensorRT include dir
include_directories(${TRT_HOME}/include)
include_directories(${TRT_HOME}/samples)

# Personal Src include 
include_directories(${CMAKE_SOURCE_DIR}/src/TrtLib)

# # CUDA library dir
set(CUDA_LIB_DIR ${CUDA_HOME}/lib64)

# TensorRT library dir
set(TRT_LIB_DIR ${TRT_HOME}/lib)

# # CUDA libs
set(LD_CUDA_LIBS cuda cudart)

# TensorRT libs
set(LD_TRT_LIBS myelin nvinfer nvinfer_plugin nvonnxparser nvparsers)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.c *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

link_directories(${CUDA_LIB_DIR}) 
link_directories(${TRT_LIB_DIR}) 

# create static library
add_library(${PROJECT_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${LD_TRT_LIBS} ${LD_CUDA_LIBS})
