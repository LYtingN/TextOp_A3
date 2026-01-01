cmake_minimum_required(VERSION 3.16)
project(unitree_mujoco)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  message("Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -fPIC")
else()
  message("Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -DNDEBUG -fPIC")
endif()

# Check if we're building with ROS2 (default OFF to avoid conflicts)
option(BUILD_ROS2 "Build ROS2 version" OFF)

# Temporarily remove conflicting ROS2 paths to avoid DDS version conflicts if not building ROS2
if(NOT BUILD_ROS2)
  list(REMOVE_ITEM CMAKE_PREFIX_PATH "/opt/ros/foxy")
  set(CMAKE_SYSTEM_IGNORE_PATH "${CMAKE_SYSTEM_IGNORE_PATH};/opt/ros/foxy/include;/opt/ros/foxy/lib")
endif()

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(mujoco REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FMT REQUIRED fmt)

# ROS2 dependencies (only if building ROS2 version)
if(BUILD_ROS2)
  # Add ROS2 to CMAKE_PREFIX_PATH if not already there
  list(APPEND CMAKE_PREFIX_PATH "/opt/ros/foxy")
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
endif()

file(GLOB SIM_SRC
    src/joystick/joystick.cc
    src/mujoco/*.cc
)

set(SIM_DEPENDENCIES
    pthread
    mujoco::mujoco
    glfw
    yaml-cpp
    unitree_sdk2
    boost_program_options
    ${FMT_LIBRARIES}
)

# Build original version
add_executable(unitree_mujoco ${SIM_SRC} src/main.cc)
target_include_directories(unitree_mujoco PRIVATE ${FMT_INCLUDE_DIRS})
target_link_libraries(unitree_mujoco ${SIM_DEPENDENCIES})


add_executable(jstest src/joystick/jstest.cc src/joystick/joystick.cc)
