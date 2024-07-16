cmake_minimum_required(VERSION 3.0.2)
project(sim2real)
set(CMAKE_CXX_STANDARD 17)
set(Torch_DIR /home/hpx/libtorch/share/cmake/Torch)
set(PythonInterp_DIR /usr/bin/python3) 
add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# Find catkin
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    livelybot_msg  
    serial
    # dynamic_reconfigure
    livelybot_serial
    sim2real_msg
)
find_package(Torch REQUIRED)

set(rt_LIBRARIES rt)
set(pthread_LIBRARIES pthread)

catkin_package(
    LIBRARIES ${PROJECT_NAME}
    INCLUDE_DIRS include
    DEPENDS rt pthread
    CATKIN_DEPENDS roscpp std_msgs livelybot_msg serial livelybot_serial sim2real_msg
)

find_library(SERIALPORT_LIBRARY serialport)
if(NOT SERIALPORT_LIBRARY)
  message(FATAL_ERROR "libserialport not found")
endif()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

add_executable(sim2real src/sim2real.cpp ${serial_SRCS})
add_dependencies(sim2real ${PROJECT_NAME})
target_link_libraries(sim2real ${catkin_LIBRARIES} ${TORCH_LIBRARIES} ${SERIALPORT_LIBRARY} yaml-cpp)
