cmake_minimum_required(VERSION 3.5)

project(shape-detection LANGUAGES CXX)

# project(testProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(SOURCES

    src/main.cpp
    src/convenience.cpp

)

add_executable(shape-detection ${SOURCES})

target_include_directories(shape-detection
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( shape-detection ${OpenCV_LIBS} )
