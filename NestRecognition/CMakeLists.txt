cmake_minimum_required(VERSION 3.0)
project(NestRecognition)

find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)
include_directories( ${OPENCV_INCLUDE_DIRS} )
include_directories( ${Caffe_INCLUDE_DIRS} )
add_definitions( ${Caffe_DEFINITIONS} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(NestRecognition ${SOURCE_FILES} Classifier.cpp Classifier.h ISlideMethod.h SlideWindowMethod.cpp SlideWindowMethod.h SelectiveMethod.cpp SelectiveMethod.h SelectiveSearchMethod/SelectiveSearchMethod.cpp SelectiveSearchMethod/SelectiveSearchMethod.h)
target_link_libraries( NestRecognition ${OpenCV_LIBS} )
target_link_libraries( NestRecognition ${Caffe_LIBRARIES} )