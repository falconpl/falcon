####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)
# Enable this include if the project has a private include
#INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include" )

#
# Falcon generic directories
#

INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

IF(WIN32)
   IF(MSVC)
      SET( COMP_FILES
         dynlib_sys_msvc.cpp )
   ELSE(MSVC)
      #Mingw -- gcc
      SET( COMP_FILES
         dynlib_sys_gcc.cpp )
   ENDIF(MSVC)

   SET( SYS_FILES
      ${COMP_FILES}
      dynlib_sys_win.cpp )

   # No other library needed on windows
   SET( SYS_LIBS )
ELSE(WIN32)
   IF(${CMAKE_SYSTEM_NAME} STREQUAL "SunOS" )
      SET( COMP_FILES
         dynlib_sys_sunc.cpp )
   ELSE(${CMAKE_SYSTEM_NAME} STREQUAL "SunOS" )
      #GCC/Xcode
      SET( COMP_FILES
         dynlib_sys_gcc.cpp )
   ENDIF(${CMAKE_SYSTEM_NAME} STREQUAL "SunOS")

   IF(APPLE)
      SET( SYS_SPECIFIC
         dynlib_sys_macosx.cpp )
   ELSE(APPLE)
      SET( SYS_SPECIFIC
         dynlib_sys_posix.cpp )
   ENDIF(APPLE)

   SET( SYS_FILES
      ${COMP_FILES}
      ${SYS_SPECIFIC}
      dynlib_sys_dl.cpp    #COMMON to all unices
      )

   # TODO: BSD has no libs and MacOSX has a different lib
   SET( SYS_LIBS
      dl )
ENDIF(WIN32)

#sources
SET( SRC_FILES
   dynlib.cpp
   dynlib_ext.cpp
   dynlib_mod.cpp
   dynlib_st.cpp

   ${SYS_FILES}
   )

# Target
ADD_LIBRARY( dynlib_fm MODULE
   ${SRC_FILES}
)

TARGET_LINK_LIBRARIES( dynlib_fm falcon_engine ${SYS_LIBS})
FALCON_INSTALL_MODULE( dynlib_fm )
