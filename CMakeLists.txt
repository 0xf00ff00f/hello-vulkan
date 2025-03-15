cmake_minimum_required(VERSION 3.29)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "a9e1cf81-9932-4810-974b-6eccaf14e457")
set(CMAKE_CXX_MODULE_STD ON)

project(hello)

add_subdirectory(glfw)

find_package(Vulkan REQUIRED)

add_executable(hello main.cc)
target_link_libraries(hello PUBLIC glfw Vulkan::Vulkan)
target_compile_features(hello PUBLIC cxx_std_23)
set_target_properties(hello PROPERTIES CXX_STANDARD_REQUIRED ON)
