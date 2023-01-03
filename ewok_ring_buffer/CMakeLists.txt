cmake_minimum_required(VERSION 2.8.3)
project(ewok_ring_buffer CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_definitions(-std=c++11 -g -O3 -ftree-vectorize -march=native)

find_package(CUDA REQUIRED)
find_package(catkin_simple REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)

set(CUDA_NVCC_FLAGS "-arch=compute_50" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC)
cuda_add_library(CUDA_Pointcloud ${LIB_TYPE} src/pointcloud_processing.cu)

include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OCTOMAP_INCLUDE_DIRS})

catkin_simple()

cs_add_executable(ring_buffer_example src/ring_buffer_example.cpp)
add_dependencies(ring_buffer_example CUDA_Pointcloud)
target_link_libraries(ring_buffer_example ${catkin_LIBRARIES} CUDA_Pointcloud)
cs_add_executable(tum_rgbd_ring_buffer_example src/tum_rgbd_ring_buffer_example.cpp)
add_dependencies(tum_rgbd_ring_buffer_example CUDA_Pointcloud)
target_link_libraries(tum_rgbd_ring_buffer_example ${OCTOMAP_LIBRARIES})

catkin_add_gtest(test_ring_buffer_base test/ring-buffer-base-test.cpp)

cs_install()
cs_export()
