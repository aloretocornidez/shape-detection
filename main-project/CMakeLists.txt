cmake_minimum_required(VERSION 3.5)


############################################
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_COLOR_MAKEFILE ON)
# set(VERBOSE_BUILD ON)
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set(CMAKE_MACOSX_RPATH TRUE)
# set_property(GLOBAL PROPERTY USE_FOLDERS ON)
###########################################



set(PROJECT_NAME "shape-detection-wrapper")

project(${PROJECT_NAME})


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(src)
