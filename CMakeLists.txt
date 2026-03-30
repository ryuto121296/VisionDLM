cmake_minimum_required(VERSION 3.21)
project(VisionDLM)

# 1. FIX: TORCH_CUDA_ARCH_LIST must be set BEFORE find_package(Torch)
# We also set CMAKE_CUDA_ARCHITECTURES to empty to stop the warning 
# and let LibTorch take full control of the GPU flags.
set(TORCH_CUDA_ARCH_LIST "7.5;8.6;8.9;12.0")
set(CMAKE_CUDA_ARCHITECTURES OFF) 

# 2. FIX: Help LibTorch find cuDNN
# Some versions of LibTorch look for CUDNN_INCLUDE_DIR and CUDNN_LIBRARY 
# rather than just the ROOT.
set(CUDNN_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/include")
set(CUDNN_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/lib/x64/cudnn.lib")
set(ENV{CUDNN_ROOT} "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")

# 3. C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Determine which LibTorch version to use based on the build type
# Note: For Visual Studio, you must set the build type during the CMake configure step 
# or via the command line: -DCMAKE_BUILD_TYPE=Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LIBTORCH_DIR "D:/_Build_CPP/LibTorch/2.11.0+cu130/debug/libtorch")
    message(STATUS "Configuring for DEBUG build using: ${LIBTORCH_DIR}")
else()
    # Default to Release if not specified or if set to Release
    set(LIBTORCH_DIR "D:/_Build_CPP/LibTorch/2.11.0+cu130/release/libtorch")
    message(STATUS "Configuring for RELEASE build using: ${LIBTORCH_DIR}")
endif()

list(APPEND CMAKE_PREFIX_PATH 
    "${LIBTORCH_DIR}"
    "C:/opencv/opencv-4.11.0/build" 
)

# --- 5. Find Packages ---
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(Torch REQUIRED) 
find_package(OpenCV REQUIRED)

# --- 6. Create Executable ---
add_executable(my_app main.cpp 
    sources/models/models.h
    sources/models/UnifiedVisionNetImpl.h
    sources/models/Resnet50_Vision.h
    sources/algorithm/model_handler.h
    sources/algorithm/dataset.h
)

# --- 7. Link All Libraries ---
target_link_libraries(my_app PRIVATE 
    Qt6::Widgets 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# --- 8. DLL Deployment (Post-Build) ---
if(MSVC)
    add_custom_command(TARGET my_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:my_app>
            $<TARGET_FILE_DIR:my_app>
        COMMAND_EXPAND_LISTS
        COMMENT "Deploying DLLs to build folder..."
    )
endif()