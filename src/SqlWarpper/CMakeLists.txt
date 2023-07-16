cmake_minimum_required(VERSION 3.15)
project(SqlWarpper)
file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.c *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

# MySql++
set(MYSQLPP_ROOT ${CMAKE_SOURCE_DIR}/3rdparty/mysql++)

include_directories(${MYSQLPP_ROOT}/include)
include_directories(/usr/include/mysql) # mysql include dirs
link_directories(${MYSQLPP_ROOT}/lib)

# create static library
add_library(${PROJECT_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(${PROJECT_NAME} mysqlclient)




