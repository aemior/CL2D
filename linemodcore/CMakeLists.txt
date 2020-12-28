cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(linemodcore)


# debug or release
SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS "-fPIC")
SET(CMAKE_C_FLAGS "-fPIC")


# arm or x86
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
ELSE()
    SET(PLATFORM_COMPILE_FLAGS "-march=native")

    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
    SET(PLATFORM_COMPILE_FLAGS "-msse -msse2 -msse3 -msse4 -mssse3")  # SSE only
ENDIF()

# SET(PLATFORM_COMPILE_FLAGS "-DMIPP_NO_INTRINSICS")  # close SIMD
SET(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")


# opencv
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/aemior/resource/opencv_install")
find_package(OpenCV 3.4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# include MIPP headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/include/MIPP/")
# include wykobi headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/include/wykobi/")
# include headers
include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/include")


# set source directory
AUX_SOURCE_DIRECTORY(src DIR_SRCS)
SET(ALL_SRC ${DIR_SRCS})

add_compile_options(-fPIC)

#Options MODE_FUSION+MODE_NMS+MODE_GRAY for CL2D
#MODE_DEBUG for time info 
#add_definitions(-D MODE_HALIDE)
add_definitions(-D MODE_FUSION)
add_definitions(-D MODE_NMS)
add_definitions(-D MODE_GRAY)
#add_definitions(-D MODE_DEBUG)

add_definitions(-D THR=50)

add_library(${PROJECT_NAME} SHARED ${ALL_SRC})
target_link_libraries(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/lib/quant_ori.a)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

