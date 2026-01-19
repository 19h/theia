cmake_minimum_required(VERSION 3.20)

project(psynth VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PSYNTH_BUILD_APPS "Build CLI applications" ON)
option(PSYNTH_BUILD_VIEWER "Build interactive 3D viewer" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc features2d calib3d flann)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# OpenMP support for parallel processing
# On macOS with Apple Clang, need to help CMake find Homebrew's libomp
if(APPLE)
    # Check for Homebrew libomp
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE HOMEBREW_LIBOMP_RESULT
    )
    if(HOMEBREW_LIBOMP_RESULT EQUAL 0 AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
        set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
        message(STATUS "Found Homebrew libomp at: ${HOMEBREW_LIBOMP_PREFIX}")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel processing")
else()
    message(WARNING "OpenMP not found - parallel processing disabled")
endif()

add_library(psynth
  src/config.cpp
  src/io/dataset.cpp
  src/io/serialization.cpp
  src/io/export.cpp
  src/features/sift.cpp
  src/matching/matcher.cpp
  src/geometry/fundamental.cpp
  src/geometry/essential.cpp
  src/geometry/triangulation.cpp
  src/geometry/pnp.cpp
  src/sfm/tracks.cpp
  src/sfm/incremental_sfm.cpp
  src/ba/bundle_adjuster.cpp
  src/mvs/pmvs_export.cpp
)

target_include_directories(psynth
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(psynth
  PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Ceres::ceres
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# Define PSYNTH_USE_OPENMP if OpenMP is available
if(OpenMP_CXX_FOUND)
    target_compile_definitions(psynth PUBLIC PSYNTH_USE_OPENMP)
endif()

target_compile_features(psynth PUBLIC cxx_std_17)

if(MSVC)
  target_compile_definitions(psynth PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(PSYNTH_BUILD_APPS)
  add_executable(psynth_pipeline apps/psynth_pipeline.cpp)
  target_link_libraries(psynth_pipeline PRIVATE psynth)
endif()

if(PSYNTH_BUILD_VIEWER)
  find_package(glfw3 3.3 REQUIRED)
  find_package(OpenGL REQUIRED)
  find_package(glm REQUIRED)
  add_subdirectory(src/viewer)
endif()