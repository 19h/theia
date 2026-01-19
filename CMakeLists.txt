cmake_minimum_required(VERSION 3.20)

project(psynth VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PSYNTH_BUILD_APPS "Build CLI applications" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc features2d calib3d flann)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

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
)

target_compile_features(psynth PUBLIC cxx_std_17)

if(MSVC)
  target_compile_definitions(psynth PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(PSYNTH_BUILD_APPS)
  add_executable(psynth_pipeline apps/psynth_pipeline.cpp)
  target_link_libraries(psynth_pipeline PRIVATE psynth)
endif()