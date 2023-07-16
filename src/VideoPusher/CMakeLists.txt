cmake_minimum_required(VERSION 3.13)

project(VideoPusher)

set(CMAKE_CXX_STANDARD 11)

# set(CMAKE_CXX_FLAGS "-g -Wall")

#--------------- FFMpeg ----------------------
set(FFMPEG_PATH " /usr/local/ffmpeg")
INCLUDE_DIRECTORIES(${FFMPEG_PATH}/include)
LINK_DIRECTORIES(${FFMPEG_PATH}/lib)

#--------------- OpenCV ----------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# #--------------- QT5 ----------------------
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/zwy/Qt5.14.2/5.14.2/gcc_64/lib/cmake")

#打开Qt的MOC功能和UI文件处理功能
# set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTOUIC ON)
# set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# find_package(Qt5Core REQUIRED)
# find_package(Qt5Widgets REQUIRED)
# find_package(Qt5Sql REQUIRED)
# find_package(Qt5Multimedia REQUIRED)
# find_package(Threads REQUIRED)
#--------------------------------------------

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.c *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

set(FFMEPG_LIB ${FFMEPG_LIB} avcodec  avdevice  avfilter  avformat  avutil  postproc  swresample  swscale)

add_library(${PROJECT_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${FFMEPG_LIB} ${OpenCV_LIBS})



# # 链接库文件
# TARGET_LINK_LIBRARIES(OpenCVFFmpegRtmp
#         avcodec  avdevice  avfilter  avformat  avutil  postproc  swresample  swscale
#         ${OpenCV_LIBS}
#         ${CMAKE_THREAD_LIBS_INIT}
#         Qt5::Core Qt5::Widgets Qt5::Sql Qt5::Multimedia
#         z
#         )