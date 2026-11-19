cmake_minimum_required(VERSION 3.18)
project(TeslaBEVHazardPerception LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

add_executable(bev_perception src/cpp/main.cpp)
target_link_libraries(bev_perception ${OpenCV_LIBS} nvinfer nvinfer_plugin cudart)
