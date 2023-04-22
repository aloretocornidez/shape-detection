cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME "shape-detection")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra -std=c++11")

# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_COLOR_MAKEFILE ON)
# set(VERBOSE_BUILD ON)
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set(CMAKE_MACOSX_RPATH TRUE)
# set_property(GLOBAL PROPERTY USE_FOLDERS ON)


project(${PROJECT_NAME} LANGUAGES CXX CUDA)






set(SOURCES
main.cpp
hough-transform.cu
hough-transform.hpp
interface.hpp
interface.cpp
)


# Creating the executable
add_executable(${PROJECT_NAME} ${SOURCES})


# Adding Computing architectures (compute capabilities)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;61")


################    OPENCV INCLUSION    ################
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
################    END OPENCV INCLUSION    ################

find_package( CUDA REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${CUDA_LIBS} )
