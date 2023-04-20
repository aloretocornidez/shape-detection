cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME "shape-detection")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra -std=c++11")

project(${PROJECT_NAME} LANGUAGES CXX)



set(SOURCES
    src/main.cpp
    src/homework-6.cpp
    src/parallel-cv.cpp

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

################    CUDA INCLUSION    ################
find_package(CUDA REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --verbose")

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
set(COMPUTE_CAPABILITY 60)
# set(CUDA_SEPARABLE_COMPILATION TRUE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --gpu-architecture;sm_${COMPUTE_CAPABILITY}")
# list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)

# Set warning levels for host compilation
if (UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra -std=c++11")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra")
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wno-unused-function,-Wno-unused-parameter")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif ()


# Add debugging to CUDA NVCC flags (for NVidia's NSight tools)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

# Set the correct linker options on UNIX platforms
# -lpthread inserted by ali akoglu  for compatiblity with elgato
if (UNIX)
    set(LINK_LIBRARIES "-lpthread -lm")
    if (APPLE)
        find_library(CUDA_LIBRARY NAMES CUDA)
        get_filename_component(FRAMEWORK_DIRECTORY ${CUDA_LIBRARY} DIRECTORY)
        set(LINK_LIBRARIES "${LINK_LIBRARIES} -F${FRAMEWORK_DIRECTORY} -framework CUDA")
    else ()
        include(CheckLibraryExists)
        check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
        if (HAVE_CLOCK_GETTIME)
            set(LINK_LIBRARIES "${LINK_LIBRARIES} -lrt")
        else ()
            message(FATAL_ERROR "clock_gettime() not found")
        endif ()
    endif ()
endif ()

################    END CUDA INCLUSION    ################
