####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib
####################################################################

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Falcon_INCLUDE_DIRS}
)

if(WIN32)
  if(MSVC)
    set(COMP_FILES dynlib_sys_msvc.cpp )
  else()
    #Mingw -- gcc
    set(COMP_FILES dynlib_sys_gcc.cpp )
  endif()

  set(SYS_FILES
    ${COMP_FILES}
    dynlib_sys_win.cpp
  )

  unset(SYS_LIBS) # No other library needed on windows
else()
  if(CMAKE_SYSTEM_NAME STREQUAL "SunOS" )
    set(COMP_FILES dynlib_sys_sunc.cpp)
  else(CMAKE_SYSTEM_NAME STREQUAL "SunOS" )
    #GCC/Xcode
    set( COMP_FILES dynlib_sys_gcc.cpp )
  endif(CMAKE_SYSTEM_NAME STREQUAL "SunOS")

  if(APPLE)
    set(SYS_SPECIFIC dynlib_sys_macosx.cpp)
    set(SYS_LIBS dl)
  else()
    set(SYS_SPECIFIC dynlib_sys_posix.cpp)
    if(NOT CMAKE_SYSTEM_NAME MATCHES ".*BSD")
      list(APPEND SYS_LIBS dl rt)
    endif()
  endif()

  set(SYS_FILES
    ${COMP_FILES}
    ${SYS_SPECIFIC}
    dynlib_sys_dl.cpp    # COMMON to all unices
  )
endif(WIN32)

#sources
set(SRC_FILES
  dynlib.cpp
  dynlib_ext.cpp
  dynlib_mod.cpp
  dynlib_st.cpp
  dynlib_parser.cpp

  ${SYS_FILES}
)

# Target
add_library(dynlib_fm MODULE
  ${SRC_FILES}
)

target_link_libraries(dynlib_fm ${SYS_LIBS})
falcon_finalize_module(dynlib_fm)
