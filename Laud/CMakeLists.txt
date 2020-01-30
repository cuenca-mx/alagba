cmake_minimum_required(VERSION 3.0)
project(Laud)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(Laud ${SOURCE_FILES})
target_link_libraries( Laud ${OpenCV_LIBS} )
