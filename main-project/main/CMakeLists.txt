cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME "homework-6")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra -std=c++11")

project(${PROJECT_NAME} LANGUAGES CXX)



set(SOURCES
    src/main.cpp
    src/homework-6.cpp

)


# Creating the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# This includes the header files that I make.
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Includes the directores of the libraries that I make.
target_link_libraries(${PROJECT_NAME}
    sub::dilation
    sub::erosion

)




################    OPENCV INCLUSION    ################
# This includes the opencv headerfiles.
find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS}/src )
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
################    END OPENCV INCLUSION    ################
