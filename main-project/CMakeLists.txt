cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME "homework-6-driver")

project(${PROJECT_NAME})


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(image-processing)
add_subdirectory(main)
