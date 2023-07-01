cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME "shape-detection")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(VERBOSE_BUILD ON)





set(SOURCES
main.cpp
hough-transform.cu
hough-transform.hpp
helper-functions.hpp
helper-functions.cpp
)


# Creating the executable
add_executable(${PROJECT_NAME} ${SOURCES})


# Adding Computing architectures (compute capabilities)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "86")


################    OPENCV INCLUSION    ################
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
################    END OPENCV INCLUSION    ################


project(${PROJECT_NAME} LANGUAGES CXX CUDA)
# find_package(CUDA REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS})
# include_directories("${CUDA_INCLUDE_DIRS}")
# target_link_libraries( ${PROJECT_NAME} ${CUDA_LIBS} )

