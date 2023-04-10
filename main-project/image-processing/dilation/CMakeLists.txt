cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME "dilation")
project(${PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    src/dilation.cpp

)

add_library(${PROJECT_NAME}
    src/dilation.cpp
)
add_library(sub::dilation ALIAS ${PROJECT_NAME})

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# add_executable(${PROJECT_NAME} ${SOURCES})


# This includes the header files that I make.
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# This includes the opencv headerfiles.
find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS}/src )
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )


