cmake_minimum_required(VERSION 3.15)

project(TRTSegServer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

#cmake在编译过程中打印编印信息
set(CMAKE_VERBOSE_MAKEFILEON ON) 
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)

find_package(OpenCV)
enable_language(CUDA)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_HOME /usr/local/cuda)
set(TRT_HOME /home/zwy/TensorRT-7.2.3.4)

# 设置动态链接库路径
set(CUDA_LIB_DIR ${CUDA_HOME}/lib64)
set(TRT_LIB_DIR ${TRT_HOME}/lib)
set(LD_CUDA_LIBS cuda cudart)
set(LD_TRT_LIBS myelin nvcaffe_parser nvinfer nvinfer_plugin nvonnxparser nvparsers)

# 设置输出bin文件路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations")
# cuda 和 cudnn 头文件
include_directories(${CUDA_HOME}/include)
include_directories(${CUDA_HOME}/targets/x86_64-linux/include)

# TensorRT 头文件
include_directories(${TRT_HOME}/include)
include_directories(${TRT_HOME}/sample)

# OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# FFMpeg 头文件
set(FFMPEG_PATH " /usr/local/ffmpeg")
INCLUDE_DIRECTORIES(${FFMPEG_PATH}/include)
LINK_DIRECTORIES(${FFMPEG_PATH}/lib)

# src 链接库
add_subdirectory(${CMAKE_SOURCE_DIR}/src/TrtLib/common)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/TrtLib/builder)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/TrtLib/infer)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/HttpServer)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/YOLOv8Seg)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/VideoPusher)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/SqlWarpper)

set(FFMEPG_LIB ${FFMEPG_LIB} avcodec  avdevice  avfilter  avformat  avutil  postproc  swresample  swscale)
set(EXTRA_LIBS ${EXTRA_LIBS} VideoPusher YOLOv8Seg HttpServer TrtInfer TrtBuilder common)

link_directories(${CUDA_LIB_DIR} ${TRT_LIB_DIR}) 

add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/main.cpp)
# 链接动态链接库
target_link_libraries(${PROJECT_NAME}  ${LD_TRT_LIBS} ${LD_CUDA_LIBS} ${OpenCV_LIBS} ${EXTRA_LIBS})

add_executable(test_yolov8_seg ${CMAKE_SOURCE_DIR}/example/test_yolov8_seg.cpp)
# 链接动态链接库
target_link_libraries(test_yolov8_seg  ${LD_TRT_LIBS} ${LD_CUDA_LIBS} ${OpenCV_LIBS} ${EXTRA_LIBS})

add_executable(test_http_server ${CMAKE_SOURCE_DIR}/example/test_http_server.cpp)
# 链接动态链接库
target_link_libraries(test_http_server  ${LD_TRT_LIBS} ${LD_CUDA_LIBS} ${OpenCV_LIBS} ${EXTRA_LIBS})

add_executable(test_video_pusher ${CMAKE_SOURCE_DIR}/example/test_video_pusher.cpp)
# 链接动态链接库
target_link_libraries(test_video_pusher  ${LD_TRT_LIBS} ${LD_CUDA_LIBS} ${OpenCV_LIBS} ${EXTRA_LIBS} ${FFMEPG_LIB})