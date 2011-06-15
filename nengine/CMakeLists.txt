##################################################
# Falcon Programming Language
#  -- New engine tests.
##################################################

cmake_minimum_required(VERSION 2.6.2)
project(Falcon)

include_directories(
  include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

configure_file(
  include/falcon/config.h.in  ${CMAKE_CURRENT_BINARY_DIR}/include/falcon/config.h
  @ONLY
)

if(${CMAKE_SYSTEM_NAME} STREQUAL "SunOS")
   add_definitions( -D_POSIX_PTHREAD_SEMANTICS )
endif()

add_subdirectory(engine)
add_subdirectory(tests)
