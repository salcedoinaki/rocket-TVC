cmake_minimum_required(VERSION 3.14)
project(RocketTVC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for embedded safety
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
    add_compile_options(-fno-exceptions)  # Embedded constraint
elseif(MSVC)
    add_compile_options(/W4 /WX)
endif()

# Library
add_library(rocket_control STATIC
    rocket_control.cpp
    rocket_control.hpp
)

# Test executable
add_executable(test_control
    test_control.cpp
)
target_link_libraries(test_control PRIVATE rocket_control)

# Enable testing
enable_testing()
add_test(NAME RocketControlTests COMMAND test_control)
