cmake_minimum_required(VERSION 3.0)
project(Tracking)

find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)
include_directories( ${OPENCV_INCLUDE_DIRS} )
include_directories( ${Caffe_INCLUDE_DIRS} )
add_definitions( ${Caffe_DEFINITIONS} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(Tracking ${SOURCE_FILES} Classifier/Classifier.cpp Classifier/Classifier.h Classifier/ISlideMethod.h Classifier/SelectiveMethod.cpp Classifier/SelectiveMethod.h Classifier/SlideWindowMethod.cpp Classifier/SlideWindowMethod.h Classifier/SelectiveSearchMethod/SelectiveSearchMethod.cpp Classifier/SelectiveSearchMethod/SelectiveSearchMethod.h)
target_link_libraries( Tracking ${OpenCV_LIBS} )
target_link_libraries( Tracking ${Caffe_LIBRARIES} )