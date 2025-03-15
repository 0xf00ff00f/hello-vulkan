cmake_minimum_required(VERSION 3.22)

project(hello)

add_subdirectory(glfw)

find_package(Vulkan REQUIRED)

add_executable(hello main.cc)
target_link_libraries(hello PUBLIC glfw Vulkan::Vulkan)
set_property(TARGET hello PROPERTY CXX_STANDARD 23)
