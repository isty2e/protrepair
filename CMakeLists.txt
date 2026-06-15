cmake_minimum_required(VERSION 3.18)

project(protrepair LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
  FASPR
  vendor/faspr/src/AAName.cpp
  vendor/faspr/src/FASPR.cpp
  vendor/faspr/src/PairEnergy.cpp
  vendor/faspr/src/RotamerBuilder.cpp
  vendor/faspr/src/Search.cpp
  vendor/faspr/src/SelfEnergy.cpp
  vendor/faspr/src/Structure.cpp
  vendor/faspr/src/Utility.cpp
)

target_include_directories(FASPR PRIVATE vendor/faspr/src)

if(NOT MSVC)
  target_compile_options(FASPR PRIVATE -ffast-math)
endif()

install(
  TARGETS FASPR
  RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}/protrepair/packing/faspr/bin"
)

install(
  FILES vendor/faspr/dun2010bbdep.bin
  DESTINATION "${SKBUILD_PLATLIB_DIR}/protrepair/packing/faspr/bin"
)
