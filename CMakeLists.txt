####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################

cmake_minimum_required(VERSION 2.6)
project(falcon-module-hpdf)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(falcon REQUIRED)
find_package(LibHaru REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${falcon_INCLUDE_DIRS}
  ${LibHaru_INCLUDE_DIRS}
)


# Target
add_library(pdf_fm MODULE
   pdf.cpp
   pdf_ext.cpp
   pdf_srv.cpp
)

target_link_libraries(pdf_fm
  ${LibHaru_LIBRARIES} 
)
falcon_finalize_module(pdf_fm)
