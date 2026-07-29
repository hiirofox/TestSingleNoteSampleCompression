cmake_minimum_required(VERSION 3.20)

project(TestLibTorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep the MSVC runtime compatible with the official LibTorch distribution.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

find_package(Torch REQUIRED)

add_executable(TestLibTorch main.cpp)
target_link_libraries(TestLibTorch PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET TestLibTorch PROPERTY
             VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
             PROPERTY VS_STARTUP_PROJECT TestLibTorch)

if(MSVC)
    target_compile_options(TestLibTorch PRIVATE /W4 /permissive-)

    # Copy all LibTorch runtime DLLs next to the executable so it can be run
    # directly from Visual Studio or from the build output directory.
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(
        TARGET TestLibTorch
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:TestLibTorch>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying LibTorch runtime DLLs"
    )
endif()
