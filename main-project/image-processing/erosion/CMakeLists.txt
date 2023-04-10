cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJECT_NAME "erosion")
project(${PROJECT_NAME})



set(SOURCES
    src/erosion.cpp
)

add_library(${PROJECT_NAME}
    ${SOURCES}
)
add_library(sub::erosion ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)


# Includes opencv files in the project
# This includes the opencv headerfiles.
find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS}/src )
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )


