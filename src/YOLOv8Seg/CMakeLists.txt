cmake_minimum_required(VERSION 3.15)
project(YOLOv8Seg LANGUAGES CUDA CXX)

# # 第三方库
# set(CUDA_HOME /usr/local/cuda)
# enable_language(CUDA)
find_package(OpenCV)
# # CUDA and cudnn include dir 
# include_directories(${CUDA_HOME}/include)
# include_directories(${CUDA_HOME}/targets/x86_64-linux/include)

# Personal Src include 
include_directories(${CMAKE_SOURCE_DIR}/src)

# OpenCV include
include_directories(${OpenCV_INCLUDE_DIRS})
# # CUDA library dir
# set(CUDA_LIB_DIR ${CUDA_HOME}/lib64)

# # CUDA libs
# set(LD_CUDA_LIBS cuda cudart)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.c *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

# link_directories(${CUDA_LIB_DIR}) 

# create static library
add_library(${PROJECT_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})